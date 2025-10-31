import os
import pandas as pd

class DataIngestionTrainingPipeline:
    def __init__(self):
        self.data_dir = "data/chest_xray"
        self.output_csv = "artifacts/image_labels.csv"

    def main(self):
        records = []
        for split in ["train", "test", "val"]:
            for label_name in ["NORMAL", "PNEUMONIA"]:
                label = 0 if label_name == "NORMAL" else 1
                folder = os.path.join(self.data_dir, split, label_name)
                for fname in os.listdir(folder):
                    if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                        records.append({
                            "split": split,
                            "path": os.path.join(folder, fname),
                            "label": label
                        })

        df = pd.DataFrame(records)
        os.makedirs("artifacts", exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        print(f"✅ Ingested {len(df)} images. Saved to {self.output_csv}")
