from IPython.display import Markdown, display

from wid_config import WIDConfig
import numpy as np

np.random.seed(42)
rng = np.random.default_rng(42)

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

data_path = Path("./data")
data_dir = data_path / "train_images"


import torch

# import torchvision.transforms as transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


torch.manual_seed(2024)


class WIDTorch:

    def __init__(
        self,
        config: WIDConfig = WIDConfig(),
        model=models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2"),
        model_name="resnet50",
        num_classes=1,
        pretrained=True,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=None,
        lr=1e-3,
        weight_decay=0.01,
        image_size=(224, 224),
    ):
        self.config = config
        self.model_dir = Path("./models") / model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 64
        self.num_workers = 4
        self.num_classes = 1
        self.epoch = 0
        self.num_epochs = 100
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = model.to(self.device)
        # self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model = self.model.to(self.device)
        self.criterion = None or nn.BCEWithLogitsLoss()
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.early_stop_threshold = 5
        self.val_losses, self.val_scores, self.train_losses = [], [], []
        self.image_size = image_size
        self.transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(
                    image_size,
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # imagenet standard
            ]
        )
        self.train_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(20),
                v2.RandomResizedCrop(
                    232,
                    scale=(0.5, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=v2.InterpolationMode.BILINEAR,
                ),
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                v2.Resize(
                    image_size,
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # models.ResNet50_Weights.IMAGENET1K_V2.transforms(),
            ]
        )
        self.val_transforms = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(
                    image_size,
                    interpolation=v2.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # imagenet standard
            ]
        )

        self.train_dataset = self.CustomDataset(
            config, wid_torch=self, df=config.df, validation=False
        )
        self.holdout_dataset = self.CustomDataset(
            config, wid_torch=self, df=config.df_holdout, validation=True
        )

        self.train_loader, self.val_loader, self.holdout_loader = self.get_dataloaders()

    def get_dataloaders(self):

        # stratified split train and validation
        train_idx, val_idx = train_test_split(
            np.arange(len(self.config.df)),
            shuffle=True,
            random_state=self.config.random_state,
            test_size=0.2,
            stratify=self.config.df["has_oilpalm"],
        )

        train_dataset = Subset(self.train_dataset, train_idx)
        val_dataset = Subset(self.train_dataset, val_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        holdout_loader = DataLoader(
            self.holdout_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        return train_loader, val_loader, holdout_loader

    def checkpoint(self, model, filename):
        torch.save(
            model.state_dict(),
            self.model_dir
            / f"Exp:{str(self.config.EXP_ID)}_Model:{self.config.model_name}_{filename}",
        )

    def resume(self, model, filename):
        model.load_state_dict(torch.load(self.model_dir / filename, weights_only=True))

    def train(self, freeze=False):
        self.epoch += 1
        running_loss, val_loss = 0.0, 0.0

        best_val_score, best_val_loss, best_epoch = None, None, None
        sigmoid = nn.Sigmoid()

        # speed things up by not require dereference of self
        model = self.model
        criterion = self.criterion
        # new optimizer each training pass
        # optimizer = self.optimizer
        optimizer = optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        train_loader = self.train_loader
        val_loader = self.val_loader
        device = self.device
        num_epochs = self.num_epochs

        if freeze:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():
                param.requires_grad = True
            num_epochs = self.epoch + freeze
        else:  # unfreeze all layers in case they were previously frozen
            for param in model.parameters():
                param.requires_grad = True
            num_epochs = self.num_epochs

        print("Training model...")

        for self.epoch in range(self.epoch, num_epochs):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            model.eval()
            with torch.inference_mode():
                val_preds, val_true = [], []

                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs.squeeze(), labels.float()).item()

                    preds = sigmoid(outputs)
                    # _, preds = torch.max(outputs.data, 1)
                    val_preds.append(preds)
                    val_true.append(labels)

                val_loss /= len(val_loader)
                running_loss /= len(train_loader)

                val_preds = torch.cat(val_preds).cpu().numpy()
                val_true = torch.cat(val_true).cpu().numpy()
                val_score = roc_auc_score(val_true, val_preds)

                # save the losses
                self.train_losses.append(running_loss)
                self.val_losses.append(val_loss)
                self.val_scores.append(val_score)

                # checkpoint model if val_score is better
                if best_val_score is None or val_score >= best_val_score:
                    best_val_score = val_score
                    best_val_loss = val_loss
                    best_epoch = self.epoch
                    self.checkpoint(
                        model,
                        f"epoch:{self.epoch}_vloss:{val_loss:.4f}_auc:{val_score:.4f}_S.pth",
                    )
                    print(f"Model saved at epoch {self.epoch} (score)")
                # also checkpoint if val_loss is lower
                elif val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = self.epoch
                    self.checkpoint(
                        model,
                        f"epoch:{self.epoch}_vloss:{val_loss:.4f}_auc:{val_score:.4f}_L.pth",
                    )
                    print(f"Model saved at epoch {self.epoch} (loss)")

                # display metrics for this epoch
                print(
                    f"Epoch [{self.epoch}/{num_epochs}], Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}, Val ROC AUC: {val_score:.4f}"
                )

                # Early stopping
                if self.epoch - best_epoch >= self.early_stop_threshold:
                    print(f"Early stopping at epoch {self.epoch}")
                    self.checkpoint(
                        model,
                        f"epoch:{self.epoch}_vloss:{val_loss:.4f}_auc:{val_score:.4f}.pth",
                    )
                    break

    def objective_function(self):
        pass  # TODO

    def plot_losses(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()

    def plot_scores(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.val_scores, label="Validation ROC AUC")
        plt.xlabel("Epoch")
        plt.ylabel("ROC AUC")
        plt.title("Validation ROC AUC")
        plt.legend()
        plt.show()

    def evaluate_holdout(self):
        model = self.model
        model.eval()
        device = self.device
        sigmoid = nn.Sigmoid()
        holdout_preds, holdout_true = [], []

        with torch.inference_mode():
            for images, labels in self.holdout_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = sigmoid(outputs)
                holdout_preds.append(preds)
                holdout_true.append(labels)

        holdout_preds = torch.cat(holdout_preds).cpu().numpy()
        holdout_true = torch.cat(holdout_true).cpu().numpy()
        holdout_score = roc_auc_score(holdout_true, holdout_preds)

        display(Markdown(f"## Holdout ROC AUC: `{holdout_score:.4f}`\n"))

        print(classification_report(holdout_true, holdout_preds >= 0.5))
        print()

        # show confusion matrix
        cm = confusion_matrix(holdout_true, holdout_preds >= 0.5)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    class CustomDataset(Dataset):
        def __init__(
            self,
            config,
            wid_torch,
            df,
            # train_transforms=None,
            # val_transforms=None,
            validation=True,
        ):
            self.config = config
            self.wid_torch = wid_torch
            self.df = df
            self.train_transforms = self.wid_torch.train_transforms
            self.val_transforms = self.wid_torch.val_transforms
            self.validation = validation

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            img_name = self.df.loc[idx, "base_path"] / self.df.loc[idx, "image_id"]
            image = Image.open(img_name).convert("RGB")
            label = torch.tensor(self.df.loc[idx, "has_oilpalm"])

            if (val_transforms := self.val_transforms) and self.validation:
                image = val_transforms(image)
            elif train_transforms := self.train_transforms:
                image = train_transforms(image)

            # weight = self.df.loc[idx, "weights"]

            return image, label  # , weight
