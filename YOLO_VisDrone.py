import locale
locale.getpreferredencoding = lambda: "UTF-8"

import torch
from loguru import logger
import wandb
from ultralytics import YOLO, checks, settings
from dotenv import dotenv_values

model_name = "yolo11n"

secrets = dotenv_values(".env")

project_name = "EyeInTheSky"
experiment_name = "YOLO_VisDrone"
dataset_name = "VisDrone"

try:
    device =  0 if torch.cuda.is_available() else "cpu"
except Exception as e:
    print(f"Error setting device: {e}")
print(f"Device: {device}")

settings.update({"wandb": True})

wandb.login(key=secrets['api_key'])

logger.info("Performing training for model...")
logger.info(checks())

model = YOLO(f"{model_name}.pt")
results = model.train(data=f"{dataset_name}.yaml",
    workers=12,
    epochs=100,
    imgsz=640,
    device=device,
    patience=5,
    project=project_name,
    name=experiment_name,
    seed=42,
    plots=True,
    val=True
  )

