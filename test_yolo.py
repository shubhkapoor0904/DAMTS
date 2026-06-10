from ultralytics import YOLO

model = YOLO("best.pt")

print("Model loaded successfully")
print(model.names)