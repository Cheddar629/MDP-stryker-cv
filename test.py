import random
import glob
from pathlib import Path
from ultralytics import YOLO

# 1. Locate your new weights
# NOTE: YOLO usually saves the first training run to 'runs/detect/train/weights/best.pt'. 
# If you ran training multiple times, it might be train2, train3, etc.
weights_path = Path("runs/detect/runs/detect/train/weights/best.pt")

if not weights_path.exists():
    print(f"❌ Could not find weights at {weights_path}.")
    print("Check your 'runs/detect/' folder to find the exact 'train' folder name.")
    exit(1)

print(f"✅ Loading newly trained model from:\n{weights_path}")
model = YOLO(weights_path)

# 2. Evaluate on the Test Set (Hard Metrics)
print("\n" + "="*40)
print("--- RUNNING EVALUATION ON TEST SET ---")
print("="*40)

metrics = model.val(
    data="d1d2_stratified/data.yaml", 
    split="test",          # Explicitly target the test split, not validation
    project="runs/detect",
    name="test_evaluation" # Saves results to runs/detect/test_evaluation
)

# Print the final scores
print("\n✅ Evaluation Complete!")
print(f"mAP@50-95 (Overall accuracy): {metrics.box.map:.4f}")
print(f"mAP@50 (Accuracy at 50% overlap): {metrics.box.map50:.4f}")

# 3. Visual Inference on Random Test Images
print("\n" + "="*40)
print("--- RUNNING VISUAL INFERENCE ---")
print("="*40)

test_images_dir = Path("d1d2_stratified/test/images")
# Grab all standard image formats
all_test_images = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    all_test_images.extend(glob.glob(str(test_images_dir / ext)))

if all_test_images:
    # Pick 5 random images so you aren't waiting forever
    sample_images = random.sample(all_test_images, min(5, len(all_test_images)))
    
    print(f"Running predictions on {len(sample_images)} random test images...")
    
    # Run prediction
    results = model.predict(
        source=sample_images,
        save=True,      # Automatically draws and saves images with bounding boxes
        conf=0.25,      # Ignore predictions with less than 25% confidence
        project="runs/detect",
        name="test_predictions" # Saves to runs/detect/test_predictions
    )
    
    print("\n✅ Visual predictions successfully saved!")
    print("Check the 'runs/detect/test_predictions' folder to view the drawn bounding boxes.")
else:
    print(f"❌ Could not find any test images in {test_images_dir}")