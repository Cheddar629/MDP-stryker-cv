from roboflow import Roboflow
from ultralytics import YOLO
import os, sys
env_bin = os.path.dirname(sys.executable)   # .../yolo-env/bin
os.environ["PATH"] = env_bin + ":" + os.environ.get("PATH", "")



# HYPERPARAMETER TUNING

# from ultralytics import YOLO
# from pathlib import Path
# import time

# # can use the non-augmented set first for cleaner signal, or swap to the augmented one if you prefer.
# # DATA_YAML = Path("/content/d1d2_aug_train_only/data.yaml")
# DATA_YAML = Path("d1d2_stratified/data.yaml")

# assert DATA_YAML.exists(), f"Missing data.yaml: {DATA_YAML}"

# model = YOLO("yolo26n.pt")

# start = time.time()


# tune_results = model.tune(
#     data=str(DATA_YAML),
#     epochs=15,
#     iterations=30,
#     imgsz=640,
#     fraction=0.5,
#     patience=5,
#     optimizer="SGD",
#     workers=2,
#     batch=16,
#     device=0,
#     project="runs/detect",
#     name="runs/detect/tune_yolo26n_tools",
#     plots=True,
#     save=True,
#     val=True,
#     verbose=True,
# )

# elapsed_min = (time.time() - start) / 60
# print(f"Tuning finished in {elapsed_min:.1f} minutes")
# print("Results saved under: /content/runs/detect/tune_yolo26n_tools")

# import shutil

# # This is where YOLO automatically puts the file based on your project/name
# tune_dir = Path("tune_yolo26n_tools")
# best_yaml = tune_dir / "best_hyperparameters.yaml"

# # Sometimes YOLO nests it in an extra 'tune' folder depending on the version
# if not best_yaml.exists():
#     best_yaml = tune_dir / "tune" / "best_hyperparameters.yaml"

# if best_yaml.exists():
#     print(f"\n✅ SUCCESS: The best hyperparameters were automatically saved at:\n{best_yaml}")
    
#     # Optional: Copy it to your current working directory so it's easier to find!
#     shutil.copy(best_yaml, "best_hyperparameters.yaml")
#     print("📂 A copy has been saved to your current folder as 'best_hyperparameters.yaml'")
# else:
#     print("❌ Could not find best_hyperparameters.yaml. Tuning may have been interrupted.")



final_model = YOLO("yolo26s.pt")

final_model.train(
    data="d1d2_aug_train_only/data.yaml",
    epochs=50,
    project="runs/detect",
    name="freeze21",
    freeze = 21 # max is 23
)

