import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        self.model_path = "artifacts/pneumonia_model.pt"
        self.dataset_path = "artifacts/transformed_datasets.pt"

    def main(self):
        # datasets = torch.load(self.dataset_path)
        datasets = torch.load(self.dataset_path, weights_only=False)
        test_loader = DataLoader(datasets["test"], batch_size=32)

        model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        y_true, y_pred = [], []
        for images, labels in test_loader:
            outputs = model(images)
            preds = torch.sigmoid(outputs).detach().numpy()
            preds = (preds > 0.5).astype(int).flatten()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

        print("✅ Accuracy:", accuracy_score(y_true, y_pred))
        print("✅ Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
        print("✅ Classification Report:\n", classification_report(y_true, y_pred))
