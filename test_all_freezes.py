import random
import glob
from pathlib import Path
from ultralytics import YOLO

# Models: (weights path, run name matching training script filename)
MODELS = [
    ("amy_runs/detect/train_freeze_1layers/weights/best.pt",  "freeze_1"),
    ("amy_runs/detect/train_freeze_3layers/weights/best.pt",  "freeze_3"),
    ("amy_runs/detect/train_freeze_5layers/weights/best.pt",  "freeze_5"),
    ("amy_runs/detect/train_freeze_7layers/weights/best.pt",  "freeze_7"),
    ("amy_runs/detect/train_freeze_9layers/weights/best.pt",  "freeze_9"),
    ("amy_runs/detect/train_freeze_backbone/weights/best.pt", "yolo26s_freezebackbone"),
]

test_images_dir = Path("d1d2_stratified/test/images")
all_test_images = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    all_test_images.extend(glob.glob(str(test_images_dir / ext)))

sample_images = random.sample(all_test_images, min(5, len(all_test_images))) if all_test_images else []

for weights_str, run_name in MODELS:
    weights_path = Path(weights_str)

    print("\n" + "="*50)
    print(f"MODEL: {run_name}")
    print("="*50)

    if not weights_path.exists():
        print(f"Could not find weights at {weights_path}. Skipping.")
        continue

    model = YOLO(weights_path)

    # 1. Evaluate on the Test Set (Hard Metrics)
    print("--- RUNNING EVALUATION ON TEST SET ---")
    metrics = model.val(
        data="d1d2_stratified/data.yaml",
        split="test",
        project="amy_runs/detect",
        name=f"test_{run_name}"
    )

    print(f"mAP@50-95 (Overall accuracy): {metrics.box.map:.4f}")
    print(f"mAP@50 (Accuracy at 50% overlap): {metrics.box.map50:.4f}")

    # 2. Visual Inference on Random Test Images
    print("--- RUNNING VISUAL INFERENCE ---")
    if sample_images:
        print(f"Running predictions on {len(sample_images)} random test images...")
        model.predict(
            source=sample_images,
            save=True,
            conf=0.25,
            project="amy_runs/detect",
            name=f"test_predictions_{run_name}"
        )
        print(f"Visual predictions saved to 'amy_runs/detect/test_predictions_{run_name}'")
    else:
        print(f"Could not find any test images in {test_images_dir}")


print("\n" + "="*50)
print("All models evaluated.")
