import os
import random
import pandas as pd
from PIL import Image
import shutil

def create_mock_dataset(base_dir, num_classes=251, num_train=100, num_val=20, num_test=20):
    """
    Creates a mock dataset structure for testing purposes.
    """

    # Create directories
    for subdir in ['train_images', 'val_images', 'test_images']:
        os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)

    # Create class_list.txt
    with open(os.path.join(base_dir, 'class_list.txt'), 'w') as f:
        for i in range(num_classes):
            f.write(f"{i} Class_{i}\n")

    # Helper to generate images and CSV
    def generate_split(split_name, num_images, has_labels=True):
        data = []
        image_dir = os.path.join(base_dir, f"{split_name}_images")

        for i in range(num_images):
            img_name = f"{split_name}_{i:05d}.jpg"
            img_path = os.path.join(image_dir, img_name)

            # Create random image
            img = Image.new('RGB', (256, 256), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            img.save(img_path)

            if has_labels:
                label = random.randint(0, num_classes - 1)
                data.append({'image_name': img_name, 'label': label})
            else:
                data.append({'image_name': img_name})

        # Save CSV
        df = pd.DataFrame(data)
        if has_labels:
            # Reorder cols just to be safe if dict order varies, though usually keys order is preserved
             df = df[['image_name', 'label']]

        df.to_csv(os.path.join(base_dir, f"{split_name}_info.csv"), index=False)
        print(f"Created {split_name} split with {num_images} images.")

    generate_split('train', num_train, has_labels=True)
    generate_split('val', num_val, has_labels=True)
    generate_split('test', num_test, has_labels=False)

if __name__ == "__main__":
    create_mock_dataset("data_mock")
