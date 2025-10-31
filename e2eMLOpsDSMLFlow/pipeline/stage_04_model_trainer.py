import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

class ModelTrainerTrainingPipeline:
    def __init__(self):
        self.dataset_path = "artifacts/transformed_datasets.pt"
        self.model_path = "artifacts/pneumonia_model.pt"
        self.batch_size = 32
        self.epochs = 5
        self.lr = 1e-4

    def main(self):
        datasets = torch.load(self.dataset_path, weights_only=False)
        train_loader = DataLoader(datasets["train"], batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(datasets["val"], batch_size=self.batch_size)

        # model = models.resnet18(pretrained=True)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        mlflow.set_experiment("PneumoniaDetection")
        with mlflow.start_run():
            for epoch in range(self.epochs):
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images = images.to("cuda" if torch.cuda.is_available() else "cpu")
                    labels = labels.float().unsqueeze(1).to("cuda" if torch.cuda.is_available() else "cpu")

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

            torch.save(model.state_dict(), self.model_path)
            mlflow.pytorch.log_model(model, "model")
            print(f"✅ Model saved to {self.model_path}")
