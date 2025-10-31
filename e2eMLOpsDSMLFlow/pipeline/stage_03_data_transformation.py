import pandas as pd
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class PneumoniaDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        max_attempts = len(self.df)
        attempts = 0
        while attempts < max_attempts:
            path = self.df.iloc[idx]["path"]
            label = self.df.iloc[idx]["label"]
            try:
                image = Image.open(path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label
            except (UnidentifiedImageError, OSError):
                print(f"❌ Skipping unreadable image: {path}")
                idx = (idx + 1) % len(self.df)
                attempts += 1
        raise RuntimeError("All images in dataset are unreadable.")
    
def is_valid_image(path):
    from PIL import Image, UnidentifiedImageError
    try:
        Image.open(path).verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


class DataTransformationTrainingPipeline:
    def __init__(self):
        self.input_csv = "artifacts/image_labels.csv"
        self.output_path = "artifacts/transformed_datasets.pt"

    def main(self):
        df = pd.read_csv(self.input_csv)

        # Filter out unreadable images
        print("🔍 Validating image files...")
        df["valid"] = df["path"].apply(is_valid_image)
        df = df[df["valid"]].drop(columns=["valid"])
        print(f"✅ Valid images retained: {len(df)}")

        # Re-split after filtering
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"
        df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

        # Print counts
        for split in ["train", "val", "test"]:
            count = df[df["split"] == split]["path"].count()
            print(f"✅ Valid images in {split}: {count}")

        # Define transforms
        base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        # Build datasets
        datasets = {}
        for split in ["train", "test", "val"]:
            split_df = df[df["split"] == split].reset_index(drop=True)
            datasets[split] = PneumoniaDataset(split_df, transform=base_transform)

        torch.save(datasets, self.output_path)
        print(f"✅ Transformed datasets saved to {self.output_path}")

