# save_model_cpu.py

import torch
from models.train_model import create_model

# Load your model
model = create_model(num_classes=7)
model.load_state_dict(torch.load("resnet50_medical.pth", map_location="mps"))

# Convert to CPU
model.cpu()

# Save new version
torch.save(model.state_dict(), "resnet50_medical_cpu.pth")

print("✅ CPU-compatible model saved as resnet50_medical_cpu.pth")
