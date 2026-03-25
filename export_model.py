from ultralytics import YOLO


model = YOLO("runs/detect/training/train2/weights/best.pt")

# 2. Export the model to ONNX format
success = model.export(format="onnx")

print("Export complete!")