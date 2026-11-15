# Localization of Image-Text Models

Final project for the Deep Learning Lab at the University of Freiburg (Summer 2023).

We extend the BLIP-2 Q-Former ([LAVIS](https://github.com/salesforce/LAVIS)) with
**localized attention**, so it can align text with a specific image region given by a
bounding box. We fine-tune it on Visual Genome and RefCOCO+ and evaluate it on COCO,
RefCOCO+, RefCOCOg and Visual Genome, including image-text retrieval.

## Structure

| Path | Contents |
|------|----------|
| `models/` | `LocalizedBlip2QFormer` and the PyTorch Lightning training module |
| `datasets/` | Dataset and data-module code for COCO, RefCOCO(+/g) and Visual Genome |
| `configs/` | Model and fine-tuning configs (localized and non-localized Q-Former) |
| `train_blip2_stage1_qformer_on_*.py` | Stage-1 Q-Former training scripts |
| `evaluate_blip2_qformer_on_*.py`, `eval_retrieval.py` | Evaluation scripts |
| `notebooks/` | Exploration, verification and retrieval-evaluation notebooks |
| `experiments/` | Saved evaluation results |

## Setup

```bash
pip install -r requirements.txt
source init_env.sh
```

The datasets aren't included. Place (or symlink) COCO 2014 at `datasets/coco2014` and
RefCOCOg at `datasets/refcocog`. Visual Genome and RefCOCO+ go in `datasets/vg` and
`datasets/refcoco+`.

## Credits

This project was built as a team by:

- **Shi Bin (Liam) Hoo**
- **Upamanyu Das**
- **Yumna Ali**
