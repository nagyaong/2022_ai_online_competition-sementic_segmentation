import os
import uuid

import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader

from transformers import AutoModelForImageClassification
from datasets import load_metric

import albumentations
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

from config import load_config
from dataset import HarborClassificationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_id = uuid.uuid4().hex[:6]

config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
train_config = config["train"]["classifier"]

model_name = config["pretrained_model_name"]

transform = albumentations.Compose([
    albumentations.GridDropout(ratio=0.2, p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.GaussNoise(p=0.2),
    albumentations.OpticalDistortion(p=0.2),
    albumentations.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
])

dataset = HarborClassificationDataset.from_config(config)
train_dataset, valid_dataset = dataset.split(valid_fraction=train_config["validation_fraction"])
train_dataset.set_transform(transform)

train_dataloader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=train_config["batch_size"])

model = AutoModelForImageClassification.from_pretrained(
    config["pretrained_model_name"], num_labels=config["num_labels"], ignore_mismatched_sizes=True
)
model.to(device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=float(train_config["learning_rate"]),
    weight_decay=float(train_config["weight_decay"])
)

losses = []
metric = load_metric("f1")

model.train()

for epoch in range(1, train_config["num_epochs"] + 1):
    for pixel_values, labels in tqdm(train_dataloader, train_id):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        metric.add_batch(
            predictions=logits.argmax(dim=-1).detach().cpu().numpy(),
            references=labels.detach().cpu().numpy()
        )

    train_f1 = metric.compute(average="micro")["f1"]

    model.eval()
    with torch.no_grad():
        for pixel_values, labels in valid_dataloader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits

            metric.add_batch(
                predictions=logits.argmax(dim=-1).detach().cpu().numpy(),
                references=labels.detach().cpu().numpy()
            )

        valid_f1 = metric.compute(average="micro")["f1"]

    model.train()

    print(
        f"epoch: {epoch}\n"
        f"├─ loss: {np.mean(losses[-100:]):.6f}\n"
        f"├─ train micro f1: {train_f1:.4f}\n"
        f"└─ valid micro f1: {valid_f1:.4f}\n"
    )

    torch.save(
        model,
        os.path.join(
            os.path.dirname(__file__),
            "checkpoints",
            f"{train_id}_{model_name.split('/')[-1]}_classifier_epoch_{epoch}.pt"
        )
    )
