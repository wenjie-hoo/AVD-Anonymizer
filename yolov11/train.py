from ultralytics import YOLO


model = YOLO("./yolo11s.pt")

model.train(
    data="pp4av.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="0",
    workers=8,
    optimizer="SGD",
    lr0=0.01,
    weight_decay=0.0005,
    name="yolov11s_finetuned",
)

metrics = model.val(data="pp4av.yaml")
print(metrics)
