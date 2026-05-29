# NYCU Computer Vision 2026 - Homework 4

Image Restoration for Rain and Snow Degradation using a PromptIR-inspired model.

## Author

Name: Cheng-Yu Cheng
Student ID: 314553035

## 1. Introduction

This project solves the HW4 image restoration task. The goal is to restore clean images from degraded images containing either rain or snow corruption.

The task requires a single model that can handle both rain and snow degradation. The model is trained from scratch using only the provided training data. No external data and no pretrained weights are used.

## 2. Method

The final model is a PromptIR-inspired image restoration network. It uses an encoder-decoder architecture with learnable prompt generation blocks. The prompt blocks learn a set of prompt tensors and adaptively combine them according to the input feature statistics.

The backbone contains restoration transformer-style blocks, including multi-DConv transposed attention and gated depthwise-convolution feed-forward modules. The model predicts a residual image, which is added to the degraded input to produce the restored output.

## 3. Key Design Choices

- Single model for both rain and snow restoration.
- PromptIR-style prompt generation modules.
- Model trained from scratch.
- No external data.
- No pretrained weights.
- Full-image tiled validation instead of center-crop validation.
- L1 loss combined with gradient loss.
- AdamW optimizer.
- Linear warmup followed by cosine learning rate decay.
- Exponential moving average of model weights for inference.

## 4. Repository Structure

NYCU-Computer-Vision-2026-HW4/
- README.md
- requirements.txt
- .gitignore
- dataset.py
- model.py
- train.py
- inference.py
- figures/

## 5. Dataset Structure

The expected dataset structure is:

release_folder/
- hw4_realse_dataset/
  - train/
    - degraded/
      - rain-1.png
      - rain-2.png
      - ...
      - rain-1600.png
      - snow-1.png
      - snow-2.png
      - ...
      - snow-1600.png
    - clean/
      - rain_clean-1.png
      - rain_clean-2.png
      - ...
      - rain_clean-1600.png
      - snow_clean-1.png
      - snow_clean-2.png
      - ...
      - snow_clean-1600.png
  - test/
    - degraded/
      - 0.png
      - 1.png
      - ...
      - 99.png

## 6. Environment Setup

Create and activate the environment:

conda create -n hw4 python=3.10 -y
conda activate hw4

Install dependencies:

pip install -r requirements.txt

## 7. Training

Single-GPU training example:

python train.py \
  --data_root ./release_folder/hw4_realse_dataset \
  --save_dir ./hw4_checkpoints_promptir_l1grad_single_b2_d48 \
  --epochs 120 \
  --warmup_epochs 15 \
  --batch_size 2 \
  --patch_size 256 \
  --steps_per_epoch 2152 \
  --dim 48 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --grad_weight 0.05 \
  --tile_size 256 \
  --overlap 64 \
  --use_ema \
  --ema_decay 0.999 \
  --num_workers 4

The best checkpoint is selected according to full-image validation PSNR.

## 8. Inference

Run inference with the best checkpoint:

python inference.py \
  --data_root ./release_folder/hw4_realse_dataset \
  --checkpoint ./hw4_checkpoints_promptir_l1grad_single_b2_d48/best_fullval_promptir.pth \
  --output_npz ./pred.npz \
  --output_zip ./submission.zip \
  --dim 48 \
  --tile_size 256 \
  --overlap 64 \
  --use_ema

The generated submission.zip contains pred.npz.

## 9. Submission Format

The pred.npz file stores restored test images in dictionary-like NumPy format.

- Key: original test filename, for example 0.png or 1.png.
- Value: restored image with shape 3 x H x W.
- Data type: uint8.

## 10. Results

The final model achieved a public CodaBench PSNR above the strong baseline.

Result summary:

Method: PromptIR-inspired model with L1 loss, gradient loss, and EMA.
Public PSNR: 30.06+

## 11. References

- PromptIR: Prompting for All-in-One Image Restoration.
- PyTorch documentation.
