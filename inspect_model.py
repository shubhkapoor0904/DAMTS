import json
from ultralytics import YOLO

try:
    model = YOLO("best2.pt")
    info = {
        "names": model.names,
        "class_count": len(model.names)
    }
    with open("model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("Successfully wrote model info to model_info.json!")
    print(info)
except Exception as e:
    print(f"Error inspecting model: {e}")
