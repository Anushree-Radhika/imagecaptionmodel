from common_imports import *
from model import tokenizer

sample_tfms = [
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.ColorJitter(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
    A.HueSaturationValue(p=0.3),
]
train_tfms = A.Compose([
    *sample_tfms,
    A.Resize(224, 224),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
    ToTensorV2(),
])
valid_tfms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
    ToTensorV2(),
])

class Dataset:
    def __init__(self, df, tfms, dir):
        self.df = df
        self.tfms = tfms
        self.dir = dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx, :]
        image = sample["image"]
        caption = sample["caption"]
        image_path = os.path.join(self.dir, image)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        augs = self.tfms(image=image)
        image = augs["image"]
        caption = f"{caption}<|endoftext|>"
        input_ids = tokenizer(caption, truncation=True)["input_ids"]
        labels = input_ids.copy()
        labels[:-1] = input_ids[1:]
        return image, input_ids, labels


def collate_fn(batch):
    image = [i[0] for i in batch]
    input_ids = [i[1] for i in batch]
    labels = [i[2] for i in batch]
    image = torch.stack(image, dim=0)
    input_ids = tokenizer.pad(
        {"input_ids": input_ids},
        padding="longest",
        return_attention_mask=False,
        return_tensors="pt",
    )["input_ids"]
    labels = tokenizer.pad(
        {"input_ids": labels},
        padding="longest",
        return_attention_mask=False,
        return_tensors="pt",
    )["input_ids"]
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels[mask == 0] = -100
    return image, input_ids, labels


def load_flickr8k():
    """Downloads Flickr8k (if not already cached) and returns (df, images_dir)."""
    import kagglehub
    path = kagglehub.dataset_download("adityajn105/flickr8k")

    images_dir = os.path.join(path, "Images")
    captions_path = os.path.join(path, "captions.txt")
    df = pd.read_csv(captions_path)

    print(f"images_dir={images_dir}")
    print(f"captions_path={captions_path}")
    print(f"df.shape={df.shape}")
    print("Total caption rows:", len(df))
    print("Unique images:", df["image"].nunique())
    print("Captions per image:\n", df.groupby("image").size().value_counts())

    return df, images_dir


def build_datasets(test_size=0.1):
    """Returns (train_ds, val_ds, train_df, val_df)."""
    df, images_dir = load_flickr8k()

    train_df, val_df = train_test_split(df, test_size=test_size)
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    # NOTE: both splits read from images_dir — the original had a bug here,
    # pointing the validation set at captions_path instead of images_dir.
    train_ds = Dataset(train_df, train_tfms, images_dir)
    val_ds = Dataset(val_df, valid_tfms, images_dir)

    print(len(train_df), len(val_df))
    return train_ds, val_ds, train_df, val_df


def get_dataloaders(batch_size=32, num_workers=2):
    """Returns (train_dl, val_dl, val_df) ready to hand to Trainer."""
    train_ds, val_ds, train_df, val_df = build_datasets()

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=True, num_workers=num_workers,
        persistent_workers=num_workers > 0, collate_fn=collate_fn,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=num_workers,
        persistent_workers=num_workers > 0, collate_fn=collate_fn,
    )
    return train_dl, val_dl, val_df

if __name__ == "__main__":
    # Quick smoke test: python dataset.py
    train_dl, val_dl, val_df = get_dataloaders(batch_size=2)
    _, c, l = next(iter(train_dl))
    print(c[0])
    print(l[0])
