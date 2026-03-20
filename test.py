import random
import glob
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# 1. Locate your new weights
weights_path = Path("runs/detect/training/train2/weights/best.pt")

if not weights_path.exists():
    print(f"❌ Could not find weights at {weights_path}.")
    print("Check your 'runs/detect/' folder to find the exact 'train' folder name.")
    import sys; sys.exit(1)

print(f"✅ Loading newly trained model from:\n{weights_path}")
model = YOLO(weights_path)

# 2. Evaluate on the Test Set
print("\n" + "="*40)
print("--- RUNNING EVALUATION ON TEST SET ---")
print("="*40)

metrics = model.val(
    data="d1d2_stratified/data.yaml", 
    split="test",          
    project="runs/detect",
    name="test_evaluation" 
)

print("\n✅ Evaluation Complete!")
print(f"mAP@50-95 (Overall accuracy): {metrics.box.map:.4f}")
print(f"mAP@50 (Accuracy at 50% overlap): {metrics.box.map50:.4f}")

# 3. Display the Generated Metrics Graphs
print("\n" + "="*40)
print("--- TEST METRICS & CONFUSION MATRIX ---")
print("="*40)

# YOLO stores the path to the saved results in metrics.save_dir
save_dir = Path(metrics.save_dir)

# Define the specific plots we want to show
plots_to_show = {
    "confusion_matrix_normalized.png": "Normalized Confusion Matrix",
    "PR_curve.png": "Precision-Recall Curve",
    "F1_curve.png": "F1-Confidence Curve"
}

for filename, title in plots_to_show.items():
    plot_path = save_dir / filename
    if plot_path.exists():
        # Read and convert the image for matplotlib
        img = cv2.imread(str(plot_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Plot it inline
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠️ Could not find {filename} in {save_dir}")

# 4. Visual Inference on Random Test Images
print("\n" + "="*40)
print("--- RUNNING VISUAL INFERENCE ---")
print("="*40)

test_images_dir = Path("d1d2_stratified/test/images")
all_test_images = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    all_test_images.extend(glob.glob(str(test_images_dir / ext)))

if all_test_images:
    sample_images = random.sample(all_test_images, min(5, len(all_test_images)))
    
    print(f"Running predictions on {len(sample_images)} random test images...")
    
    results = model.predict(
        source=sample_images,
        save=True,      
        conf=0.25,      
        project="testing_results",
        name="test_predictions" 
    )
    
    print("\n✅ Visual predictions successfully saved!")
    
    # Optional: Display the visual predictions inline too
    plt.figure(figsize=(20, 8))
    for i, r in enumerate(results):
        img_bgr = r.plot()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(results), i + 1)
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title(f"Prediction {i+1}")
    plt.tight_layout()
    plt.show()
else:
    print(f"❌ Could not find any test images in {test_images_dir}")