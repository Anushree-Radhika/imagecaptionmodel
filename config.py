from common_imports import SimpleNamespace, Path, torch

model_config = SimpleNamespace(
    vocab_size=50_257,
    embed_dim=768,
    num_heads=12,
    seq_len=1024,
    depth=12,
    attention_dropout=0.1,
    residual_dropout=0.1,
    mlp_ratio=4,
    mlp_dropout=0.1,
    emb_dropout=0.1,
)

train_config = SimpleNamespace(
    epochs=5,
    freeze_epochs_gpt=1,   # epoch at which GPT-2's own layers unfreeze
    freeze_epochs_all=2,   # epoch at which the ViT encoder also unfreezes
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_path=Path("captioner"),
    batch_size=32,
)
