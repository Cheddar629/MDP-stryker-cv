# HYPERPARAMETER TUNING

from ultralytics import YOLO
from pathlib import Path
import time

# can use the non-augmented set first for cleaner signal, or swap to the augmented one if you prefer.
# DATA_YAML = Path("/content/d1d2_aug_train_only/data.yaml")
DATA_YAML = Path("d1d2_stratified/data.yaml")

assert DATA_YAML.exists(), f"Missing data.yaml: {DATA_YAML}"

model = YOLO("yolo26s.pt")

start = time.time()


tune_results = model.tune(
    data=str(DATA_YAML),
    epochs=40,
    iterations=30,
    imgsz=640,
    patience=15,
    optimizer="auto",
    workers=2,
    batch=16,
    device='cpu',
    project="tuning_results",
    name="tune_yolo26s_results",
    plots=True,
    save=True,
    val=True,
    verbose=True,
)

elapsed_min = (time.time() - start) / 60
print(f"Tuning finished in {elapsed_min:.1f} minutes")
print("Results saved under: /tuning_results/tune_yolo26s_results")
