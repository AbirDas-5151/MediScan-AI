import torch
from torchvision import transforms
from models.train_model import create_model

CLASS_NAMES = [
    "Actinic keratoses (akiec)",
    "Basal cell carcinoma (bcc)",
    "Benign keratosis-like lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanocytic nevi (nv)",
    "Melanoma (mel)",
    "Vascular lesions (vasc)"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = create_model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load("/Users/user/Desktop/Mediscan/models/resnet50_medical_cpu.pth", map_location="cpu"))
    model.to(device)
    model.eval()
    return model

def predict(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        top_idx = torch.argmax(probs).item()
    return CLASS_NAMES[top_idx], probs[top_idx].item()
