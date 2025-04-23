from utils.dataset_loader import HAM10000Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from models.train_model import create_model, train

num_classes = len(dataset.label_map)
model = create_model(num_classes)
train(model, train_loader, val_loader, num_epochs=10)


# Setup paths
csv_path = '/Users/yourname/Downloads/Medidata/HAM10000_metadata.csv'
image_dirs = [
    '/Users/yourname/Downloads/Medidata/HAM10000_images_part_1',
    '/Users/yourname/Downloads/Medidata/HAM10000_images_part_2'
]

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
dataset = HAM10000Dataset(csv_file=csv_path, image_dirs=image_dirs, transform=transform)

# Train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
