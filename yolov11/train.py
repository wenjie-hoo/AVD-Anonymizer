from ultralytics import YOLO

# Initialize model (from scratch)
model = YOLO("./yolo11s.pt")  # You can use "yolov8s.yaml" or another config

# Train the model
model.train(
    data="pp4av.yaml",  # Path to your dataset config
    epochs=100,
    imgsz=640,
    batch=16,  # Adjust based on your GPU memory
    device="cpu",  # Use "cpu" if you don't have a GPU
    workers=8,
    optimizer="SGD",  # Try "AdamW" for better results
    lr0=0.01,  # Initial learning rate
    weight_decay=0.0005,
)
