# utils/gradcam.py

import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

def apply_gradcam(model, image_path, class_idx=None):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device, dtype=torch.float32)
    input_tensor.requires_grad_()  # Force gradient tracking on input

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    target_layer = model.layer4[-1]
    print("Hooking into:", target_layer)
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)
    pred_class = class_idx if class_idx is not None else output.argmax(dim=1).item()
    print(f"Predicted class ID: {pred_class}")

    model.zero_grad()
    score = output[0, pred_class]

    # Ensure score has gradients
    if not score.requires_grad:
        score = score.clone().detach().requires_grad_()

    # Backward pass
    score.backward()

    fh.remove()
    bh.remove()

    if not activations or not gradients:
        raise ValueError("Gradients or activations were not captured. Check hook setup.")

    act = activations[0].squeeze()
    grad = gradients[0].squeeze()

    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.sum(weights * act, dim=0)
    cam = np.maximum(cam.cpu().numpy(), 0)
    if cam.max() != 0:
        cam /= cam.max()

    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    img_np = np.array(image.resize((224, 224))).astype(np.float32) / 255

    superimposed_img = heatmap + img_np
    superimposed_img = np.clip(superimposed_img, 0, 1)

    return superimposed_img, pred_class
