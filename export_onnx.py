from ultralytics import YOLO

try:
    print("Loading best2.pt...")
    model = YOLO("best2.pt")
    print("Exporting model to ONNX format...")
    # Export with standard configuration
    model.export(format="onnx")
    print("Export complete! Saved as best2.onnx in the root directory.")
except Exception as e:
    print(f"Error during export: {e}")
