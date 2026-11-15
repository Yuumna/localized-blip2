import time
from datetime import datetime

from tqdm import tqdm
import torch
from torch.nn import functional as F
import numpy as np

from lavis.models.blip2_models.blip2_qformer import Blip2Qformer

from model_utils import concat_all_gather, all_gather_with_grad


class LocalizedBlip2QFormer(Blip2Qformer):
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        is_localized_attention=True,
    ):
        self.img_size = img_size
        self.is_localized_attention = is_localized_attention
        super().__init__(vit_model, img_size, drop_path_rate, use_grad_checkpoint,
                         vit_precision, freeze_vit, num_query_token,
                         cross_attention_freq, embed_dim, max_txt_len)
        
    @classmethod
    def from_config(cls, cfg):
        model = super().from_config(cfg)
        model.is_localized_attention = cfg.is_localized_attention
        return model
        
        
    def forward(self, samples):
        image = samples["image"]
        text = samples["text_input"]

        image_embeds = self.ln_vision(self.visual_encoder(image))
        if self.is_localized_attention:
            image_atts = create_attention_mask(image_embeds, samples["bbox_normalized"], image.device)
        else:
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )

        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_feat = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(
            image_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

        sim_q2t = torch.matmul(
            image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        ).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        # rank = dist.get_rank()
        rank = 0
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )

        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        ###============== Image-text Matching ===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        with torch.no_grad():
            weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
            weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
            weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
            weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        if self.is_localized_attention:
            image_atts_all = torch.cat(
                [image_atts, torch.ones(image_embeds_neg.size()[:-1], dtype=torch.long).to(image.device), image_atts],
                dim=0,
            )
        else:
            image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(image.device)

        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss

        return {
            "loss": loss_itc + loss_itm + loss_lm,
            "loss_itc": loss_itc,
            "loss_itm": loss_itm,
            "loss_lm": loss_lm,
        }

    def forward_image(self, image, bbox_normalized=None):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        if self.is_localized_attention and bbox_normalized is not None:
            image_atts = create_attention_mask(image_embeds, bbox_normalized, image.device)
        else:
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,

        # debug purpose
        zero_attention=False,
        return_attention_masks=False,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        # TODO not sure if this is right, but this is needed for now to work
        use_nucleus_sampling = True

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        
        if self.is_localized_attention:
            image_atts = create_attention_mask(image_embeds, samples["bbox_normalized"], image.device)
        else:
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # Debug purpose (overriding image embeddings and their attention masks)
        if zero_attention:

            # Remove the info of last token from the image embeddings
            image_embeds[:, -1] = torch.zeros_like(image_embeds[:, -1])

            # Create a mask that emphasizes the last token
            image_atts = torch.zeros(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            image_atts[:, -1] = 1

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        if return_attention_masks:
            return captions, image_atts
        else:
            return captions

    @torch.no_grad()
    def compute_sim_matrix(self, all_images, all_texts, all_bbox_info, topk=128):
        header = "Evaluation:"

        print("Computing features for evaluation...")

        # Compute text embeddings --------------------------
        print("Computing text embeddings...")
        all_text_ids, all_text_embeds, all_text_atts = [], [], []
        text_bs = 256
        num_text = len(all_texts)
        for i in range(0, num_text, text_bs):
            texts = all_texts[i : min(num_text, i + text_bs)]
            text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(self.device)
            text_feats = self.forward_text(text_inputs)
            text_embeds = F.normalize(self.text_proj(text_feats), dim=-1)

            all_text_embeds.append(text_embeds.cpu())
            all_text_ids.append(text_inputs.input_ids.cpu())
            all_text_atts.append(text_inputs.attention_mask.cpu())

        # Compute image embeddings -------------------------
        print("Computing image embeddings...")
        all_vit_feats, all_image_embeds = [], []
        all_bboxes, all_bboxes_to_img_idx = all_bbox_info
        image_bs = 256
        num_bboxes = len(all_bboxes)
        for i in range(0, num_bboxes, image_bs):
            bboxes = all_bboxes[i : min(num_bboxes, i + image_bs)].to(self.device)
            images = all_images[all_bboxes_to_img_idx[i : min(num_bboxes, i + image_bs)]].to(self.device)
            image_feats, vit_feats = self.forward_image(images, bboxes)
            image_embeds = F.normalize(self.vision_proj(image_feats), dim=-1)
            all_vit_feats.append(vit_feats.cpu())
            all_image_embeds.append(image_embeds.cpu())

        # Concatenate all embeddings -----------------------
        all_text_embeds = torch.cat(all_text_embeds, dim=0).to(self.device)
        all_text_ids = torch.cat(all_text_ids, dim=0).to(self.device)
        all_text_atts = torch.cat(all_text_atts, dim=0).to(self.device)
        all_vit_feats = torch.cat(all_vit_feats, dim=0).to(self.device)
        all_image_embeds = torch.cat(all_image_embeds, dim=0).to(self.device)
        
        print("Done generating embeddings")

        # Compute similarity matrix ------------------------
        sims_matrix = []
        for image_embed in all_image_embeds:
            sim_q2t = image_embed @ all_text_embeds.t()
            sim_i2t, _ = sim_q2t.max(0)
            sims_matrix.append(sim_i2t)
        sims_matrix = torch.stack(sims_matrix, dim=0)

        # sim_q2t = torch.matmul(all_image_embeds, all_text_embeds.t())
        # sim_i2t, _ = torch.max(sim_q2t, dim=1)
        # sims_matrix = sim_i2t.unsqueeze(1)

        score_matrix_i2t = torch.full(
            (len(all_image_embeds), len(all_text_embeds)), -100.0
        ).to(self.device)

        num_tasks = 1
        rank = 0
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            tqdm(sims_matrix[start:end], desc=f"{header} Computing similarity matrix")
        ):
            topk_sim, topk_idx = sims.topk(k=topk, dim=0)
            image_inputs = all_vit_feats[start + i].repeat(topk, 1, 1).to(self.device)
            score = self.compute_itm(
                image_inputs=image_inputs,
                text_ids=all_text_ids[topk_idx],
                text_atts=all_text_atts[topk_idx],
            ).float()
            score_matrix_i2t[start + i, topk_idx] = score + topk_sim

        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full(
            (len(all_text_embeds), len(all_image_embeds)), -100.0
        ).to(self.device)

        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            tqdm(sims_matrix[start:end], desc=f"{header} Computing similarity matrix")
        ):
            topk_sim, topk_idx = sims.topk(k=topk, dim=0)
            image_inputs = all_vit_feats[topk_idx.cpu()].to(self.device)
            score = self.compute_itm(
                image_inputs=image_inputs,
                text_ids=all_text_ids[start + i].repeat(topk, 1),
                text_atts=all_text_atts[start + i].repeat(topk, 1),
            ).float()
            score_matrix_t2i[start + i, topk_idx] = score + topk_sim

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

    @torch.no_grad()
    def compute_full_image_sim_matrix(self, all_images, all_texts, topk=128):
        header = "Evaluation:"

        print("Computing features for evaluation...")

        # Compute text embeddings --------------------------
        print("Computing text embeddings...")
        all_text_ids, all_text_embeds, all_text_atts = [], [], []
        text_bs = 256
        num_text = len(all_texts)
        for i in range(0, num_text, text_bs):
            texts = all_texts[i : min(num_text, i + text_bs)]
            text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(self.device)
            text_feats = self.forward_text(text_inputs)
            text_embeds = F.normalize(self.text_proj(text_feats), dim=-1)

            all_text_embeds.append(text_embeds.cpu())
            all_text_ids.append(text_inputs.input_ids.cpu())
            all_text_atts.append(text_inputs.attention_mask.cpu())

        # Compute image embeddings -------------------------
        print("Computing image embeddings...")
        all_vit_feats, all_image_embeds = [], []
        image_bs = 256
        num_images = len(all_images)
        for i in range(0, num_images, image_bs):
            images = all_images[i : min(num_images, i + image_bs)].to(self.device)
            image_feats, vit_feats = self.forward_image(images)
            image_embeds = F.normalize(self.vision_proj(image_feats), dim=-1)
            all_vit_feats.append(vit_feats.cpu())
            all_image_embeds.append(image_embeds.cpu())

        # Concatenate all embeddings -----------------------
        all_text_embeds = torch.cat(all_text_embeds, dim=0).to(self.device)
        all_text_ids = torch.cat(all_text_ids, dim=0).to(self.device)
        all_text_atts = torch.cat(all_text_atts, dim=0).to(self.device)
        all_vit_feats = torch.cat(all_vit_feats, dim=0).to(self.device)
        all_image_embeds = torch.cat(all_image_embeds, dim=0).to(self.device)
        
        print("Done generating embeddings")

        # Compute similarity matrix ------------------------
        sims_matrix = []
        for image_embed in all_image_embeds:
            sim_q2t = image_embed @ all_text_embeds.t()
            sim_i2t, _ = sim_q2t.max(0)
            sims_matrix.append(sim_i2t)
        sims_matrix = torch.stack(sims_matrix, dim=0)

        # sim_q2t = torch.matmul(all_image_embeds, all_text_embeds.t())
        # sim_i2t, _ = torch.max(sim_q2t, dim=1)
        # sims_matrix = sim_i2t.unsqueeze(1)

        score_matrix_i2t = torch.full(
            (len(all_image_embeds), len(all_text_embeds)), -100.0
        ).to(self.device)

        num_tasks = 1
        rank = 0
        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            tqdm(sims_matrix[start:end], desc=f"{header} Computing similarity matrix")
        ):
            topk_sim, topk_idx = sims.topk(k=topk, dim=0)
            image_inputs = all_vit_feats[start + i].repeat(topk, 1, 1).to(self.device)
            score = self.compute_itm(
                image_inputs=image_inputs,
                text_ids=all_text_ids[topk_idx],
                text_atts=all_text_atts[topk_idx],
            ).float()
            score_matrix_i2t[start + i, topk_idx] = score + topk_sim

        sims_matrix = sims_matrix.t()
        score_matrix_t2i = torch.full(
            (len(all_text_embeds), len(all_image_embeds)), -100.0
        ).to(self.device)

        step = sims_matrix.size(0) // num_tasks + 1
        start = rank * step
        end = min(sims_matrix.size(0), start + step)

        for i, sims in enumerate(
            tqdm(sims_matrix[start:end], desc=f"{header} Computing similarity matrix")
        ):
            topk_sim, topk_idx = sims.topk(k=topk, dim=0)
            image_inputs = all_vit_feats[topk_idx.cpu()].to(self.device)
            score = self.compute_itm(
                image_inputs=image_inputs,
                text_ids=all_text_ids[start + i].repeat(topk, 1),
                text_atts=all_text_atts[start + i].repeat(topk, 1),
            ).float()
            score_matrix_t2i[start + i, topk_idx] = score + topk_sim

        return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


def create_attention_mask(image_embeds, norm_bboxes, device):
    image_feats = image_embeds[:, 1:, :]
    image_feats_atts = get_attention_mask_based_on_norm_bbox(image_feats, norm_bboxes)

    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
    image_atts[:, 1:] = image_feats_atts

    return image_atts

def get_attention_mask_based_on_norm_bbox(image_embeds, norm_bboxes):
    """
    Args:
        image_embeds (torch.Tensor): A tensor of shape (batch_size, num_patches, embed_dim).
        boxes (torch.Tensor): A tensor of shape (batch_size, 4).
    Returns:
        image_att_masks (torch.Tensor): A tensor of shape (batch_size, num_patches).
    """
    batch_size, num_patches, _ = image_embeds.shape
    
    # Assuming we are dealing with square images
    assert np.sqrt(num_patches) % 1 == 0    # Assert num_patches is a perfect square
    width = int(np.sqrt(num_patches))

    # Transform normalized bounding boxes to absolute coordinates (in patches)
    boxes = norm_bboxes * width

    # Extract individual box coordinates
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # Calculate box boundaries
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # Round the box coordinates to the nearest integer
    x1 = x1.floor().to(torch.long)
    y1 = y1.floor().to(torch.long)
    x2 = x2.ceil().to(torch.long)
    y2 = y2.ceil().to(torch.long)

    # Create the image attention masks tensor
    image_att_masks = torch.zeros((batch_size, width, width), dtype=torch.long, device=image_embeds.device)

    for i in range(batch_size):
        # Set the corresponding region within the mask to 1
        image_att_masks[i, y1[i]:y2[i], x1[i]:x2[i]] = 1

    return image_att_masks.view(batch_size, num_patches)
