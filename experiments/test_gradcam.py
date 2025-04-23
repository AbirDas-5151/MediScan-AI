from torchvision import models
from utils.gradcam import apply_gradcam
import matplotlib.pyplot as plt
from models.train_model import create_model
import torch

# 👇 Replace this with your actual number of skin condition classes
num_classes = 7  # HAM10000 has 7 unique classes (akiec, bcc, bkl, df, nv, mel, vasc)

# Load the model
model = create_model(num_classes)
model.load_state_dict(torch.load("resnet50_medical.pth", map_location="mps"))

# Choose an image to test
image_path = "/Users/user/Downloads/Medidata/HAM10000_images_part_1/ISIC_0027419.jpg"

# Apply Grad-CAM
heatmap_img, pred_class = apply_gradcam(model, image_path)

# Show result
plt.imshow(heatmap_img)
plt.title(f"Predicted Class ID: {pred_class}")
plt.axis("off")
plt.show()
