# Image Captioning Model

A vision-to-language captioning model that pairs a pretrained Vision Transformer (ViT) encoder with a GPT-2-style transformer decoder, connected via cross-attention. Trained on Flickr8k.

## Architecture

- **Encoder**: `vit_base_patch16_224` (via `timm`), pretrained. Only the patch embedding, positional embedding, and transformer blocks are reused. Images are split into 16x16 patches and encoded into a sequence of patch tokens.
- **Decoder**: A GPT-2-style transformer, reimplemented from scratch (`GPT2Attention`, `GPT2CrossAttention`, `GPT2MLP`, `GPT2Block`), with pretrained GPT-2 weights loaded in via a custom `from_pretrained` method that correctly transposes the Conv1D-style weight matrices GPT-2 uses internally.
- **Fusion**: Each decoder block does causal self-attention over the caption tokens generated so far, then cross-attention where the caption tokens (query) attend over the encoder's patch tokens (key/value). This lets the decoder ground each generated word in the image's visual features rather than only the text so far.
- **Output head**: A linear LM head tied to the token embedding weights, producing next-token logits over the GPT-2 vocabulary (50,257 tokens).

## Training strategy

Fine-tuning follows a staged unfreezing schedule rather than training everything at once:

1. **Epoch 0**: Only the newly-added cross-attention layers are trainable. Both the pretrained ViT and pretrained GPT-2 weights stay frozen.
2. **After `freeze_epochs_gpt`**: GPT-2's own layers (self-attention, MLP, layer norms) are unfrozen and start fine-tuning.
3. **After `freeze_epochs_all`**: Every layer, including the ViT encoder, becomes trainable.

This lets the model first learn how to route visual information into the language model before disturbing either pretrained network's weights.

Other training details:
- Mixed-precision training (`torch.cuda.amp` autocast + GradScaler)
- OneCycleLR learning rate schedule
- Best checkpoint saved by validation perplexity
- Sampling supports temperature-based

## Dataset

[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — ~8,000 images, 5 captions each, downloaded via `kagglehub`. Images are resized to 224x224, with augmentation (horizontal flip, color jitter, brightness/contrast, shift-scale-rotate, hue/saturation) applied on the training split only.

## Files

| File | Purpose |
|---|---|
| `model.py` | `VisionGPT2Model` — encoder, decoder, cross-attention, weight loading, generation |
| `dataset.py` | Flickr8k loading, augmentation pipelines, `Dataset` class, collate function |
| `train.py` | `Trainer` class — training loop, validation, checkpointing, caption generation demo |

## Setup

```bash
pip install -r requirements.txt
```
