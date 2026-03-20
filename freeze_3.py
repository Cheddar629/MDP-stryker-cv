from ultralytics import YOLO

model = YOLO("yolo26s.pt")

model.train(
    data="d1d2_aug_train_only/data.yaml",
    epochs=50,
    project="amy_runs/detect",
    name="train_freeze_3layers",
    freeze=3,
)
