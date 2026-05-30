from ultralytics import YOLO
import os

# dataset path
DATA_YAML = os.path.join(os.path.dirname(__file__), 'garbage_classification', 'data.yaml')

if not os.path.exists(DATA_YAML):
    print(f" data.yaml not found")
    exit(1)

model = YOLO('yolov8n.pt')

results = model.train(
    data=DATA_YAML,
    epochs=30,
    imgsz=416,
    batch=16,
    device='cpu',
    workers=0,
    project='trained_model',
    name='waste_detector',
    exist_ok=True,
    verbose=True
)

print("\n Training complete!")
print(f"   Model: trained_model/waste_detector/weights/best.pt")