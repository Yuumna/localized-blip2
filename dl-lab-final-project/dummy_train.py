import os

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from models.dummy_mlp import LitModel as M
from models.dummy_mlp_lightning import LitModel
from datasets.mnist_data_module import MNISTDataModule

from training_utils import LogLastTrainingStep, make_training_dir

training_name = "train_dummy_mlp"
ckpt_path = None
# ckpt_path = f"checkpoints/{training_name}/5/last_step.ckpt"

def main():
    # model
    raw_model = M()
    model = LitModel(raw_model)
    data_loader = MNISTDataModule()

    # create subdirectory s.t. different runs don't overwrite each other
    if ckpt_path is None:
        ckpt_dir = make_training_dir("checkpoints", training_name)
    else:
        ckpt_dir = os.path.dirname(ckpt_path)

    # configure checkpointing behavior
    save_ckpt_at_epoch_end = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{step:06d}-{val_loss:.4f}",
        monitor="val_loss",
        save_last=True,
        save_on_train_epoch_end=True,
    )
    save_ckpt_at_epoch_end.CHECKPOINT_NAME_LAST = "last_epoch"
    update_last_ckpt_every_n_step = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-{step:06d}",
        save_last=True,
        every_n_train_steps=20,        
    )
    update_last_ckpt_every_n_step.CHECKPOINT_NAME_LAST = "last_step"
    log_last_training_step = LogLastTrainingStep(training_name, ckpt_dir)

    # train model
    trainer = pl.Trainer(
        accelerator="auto",
        max_epochs=5,
        callbacks=[save_ckpt_at_epoch_end, update_last_ckpt_every_n_step, log_last_training_step],
        limit_train_batches=1.0,
    )
    trainer.fit(model=model, train_dataloaders=data_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
