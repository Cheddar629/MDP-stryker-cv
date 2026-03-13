import shutil
from pathlib import Path
from ultralytics import YOLO

# 1. Point to the correct runs/detect directory
tune_dir = Path("runs/detect/runs/detect/runs/detect/tune_yolo26n_tools")
best_yaml = tune_dir / "best_hyperparameters.yaml"

# 2. Check alternative nested folders just in case
if not best_yaml.exists():
    best_yaml = tune_dir / "tune" / "best_hyperparameters.yaml"
if not best_yaml.exists():
    best_yaml = Path("runs/detect/tune/best_hyperparameters.yaml")

# 3. Only train if we actually found the file!
if best_yaml.exists():
    print(f"\n✅ SUCCESS: The best hyperparameters were found at:\n{best_yaml}")
    
    shutil.copy(best_yaml, "best_hyperparameters.yaml")
    print("📂 A copy has been saved to your current folder as 'best_hyperparameters.yaml'")
    
    # Start training using the copied file
    final_model = YOLO("yolo26n.pt")
    final_model.train(
        data="d1d2_stratified/data.yaml",
        epochs=100,
        cfg="best_hyperparameters.yaml", 
        project="runs/detect",
    )
else:
    print("❌ Could not find best_hyperparameters.yaml.")
    print("Please check your 'runs/detect/' folder manually to see where YOLO saved it.")