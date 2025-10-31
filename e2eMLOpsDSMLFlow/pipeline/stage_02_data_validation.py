import pandas as pd
import os
from PIL import Image

class DataValidationTrainingPipeline:
    def __init__(self):
        self.input_csv = "artifacts/image_labels.csv"

    def main(self):
        df = pd.read_csv(self.input_csv)

        # Check image readability
        unreadable = []
        for path in df["path"]:
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                unreadable.append(path)

        if unreadable:
            print(f"❌ Found {len(unreadable)} unreadable images:")
            for bad in unreadable[:10]:  # Show only first 10
                print(f" - {bad}")
        else:
            print("✅ All images are readable.")

        # Check class balance
        print("\n✅ Class distribution:")
        class_counts = df["label"].value_counts()
        for label, count in class_counts.items():
            label_name = "NORMAL" if label == 0 else "PNEUMONIA"
            print(f" - {label_name}: {count} images")

        # Check split distribution
        print("\n✅ Split distribution:")
        split_counts = df["split"].value_counts()
        for split, count in split_counts.items():
            print(f" - {split}: {count} images")

        print("\n✅ Data validation complete.")
