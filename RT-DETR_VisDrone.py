import locale
locale.getpreferredencoding = lambda: "UTF-8"

import torch
from loguru import logger
import wandb
from ultralytics import RTDETR, checks, settings
from dotenv import dotenv_values

model_name = "rtdetr-l.pt"
secrets = dotenv_values(".env")

project_name = "EyeInTheSky"
experiment_name = "RT-DETR_VisDrone"
dataset_name = "VisDrone"

try:
    device =  0 if torch.cuda.is_available() else "cpu"
except Exception as e:
    print(f"Error setting device: {e}")
print(f"Device: {device}")

settings.update({"wandb": True})

wandb.login(key=secrets['WANDB_API_KEY'])

logger.info("Performing training for model...")
logger.info(checks())

model = RTDETR(model_name)
for k, v in model.named_parameters():
    print(k)

model.info()
results = model.train(data=f"{dataset_name}.yaml",
    workers=1,    
    batch=4,
    epochs=100,
    half=True,
    augment=False,
    cache=False,
    amp=True,
    imgsz=640,
    device=device,
    patience=5,
    project=project_name,
    name=experiment_name,
    seed=42,
    plots=True,
    save=True,
    save_period=5
  )