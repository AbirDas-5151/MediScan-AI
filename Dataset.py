# Example using Pillow for resizing (conceptual)
from PIL import Image
import os
import pandas as pd

metadata = pd.read_csv("path/to/your/consolidated_metadata.csv")
input_dir = "data/raw/" # Adjust path based on where raw images are
output_dir = "data/processed/images_224/"
target_size = (224, 224)

os.makedirs(output_dir, exist_ok=True)

for index, row in metadata.iterrows():
    try:
        img_path = os.path.join(input_dir, row['original_path']) # You need original_path in metadata
        img = Image.open(img_path).convert('RGB') # Ensure RGB
        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
        save_path = os.path.join(output_dir, os.path.basename(row['image_path'])) # Use a unique ID or filename
        img_resized.save(save_path)
        # Update metadata with processed path if needed
    except Exception as e:
        print(f"Error processing {row['image_path']}: {e}")
        # Handle errors (e.g., remove row from metadata)


from sklearn.model_selection import train_test_split

metadata = pd.read_csv("data/processed/metadata.csv") # Assuming it has 'image_path' and 'label'
labels = metadata['label']

# Split into train+val and test
train_val_df, test_df = train_test_split(
    metadata, test_size=0.15, random_state=42, stratify=labels
)

# Split train+val into train and val
train_df, val_df = train_test_split(
    train_val_df, test_size=0.176, random_state=42, stratify=train_val_df['label'] # 0.15 / (1 - 0.15) = 0.176
)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
# Save these dataframes
train_df.to_csv("data/processed/train_metadata.csv", index=False)
val_df.to_csv("data/processed/val_metadata.csv", index=False)
test_df.to_csv("data/processed/test_metadata.csv", index=False)