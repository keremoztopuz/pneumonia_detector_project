import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

# training transforms
train_transforms = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD),
    transforms.RandomErasing(p=0.2)
])

# validation transforms
val_transforms = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.MEAN, std=config.STD)
])
    
class PneumoniaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]

def load_data(DATA_DIR):
    images = []
    labels = []
    
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory {DATA_DIR} not found")
    
    for class_name in config.CLASS_NAMES:
        class_path = os.path.join(DATA_DIR, class_name)
        label = config.CLASS_NAMES.index(class_name)
        if not os.path.exists(class_path):
            continue
        
        for file in os.listdir(class_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(os.path.join(class_path, file))
                labels.append(label)
    
    if len(images) == 0:
        raise ValueError(f"No images found in {DATA_DIR}")
    
    return images, labels
    
def create_dataloaders(batch_size=None):
    batch_size = batch_size or config.BATCH_SIZE

    train_images, train_labels = load_data(config.TRAIN_DIR)
    val_images, val_labels = load_data(config.VAL_DIR)
    test_images, test_labels = load_data(config.TEST_DIR)

    train_dataset = PneumoniaDataset(train_images, train_labels, transform=train_transforms)
    val_dataset = PneumoniaDataset(val_images, val_labels, transform=val_transforms)
    test_dataset = PneumoniaDataset(test_images, test_labels, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader
    
if __name__ == "__main__":
    try:
        train_loader, val_loader, test_loader = create_dataloaders()
        print(f"train: {len(train_loader.dataset)}")
        print(f"val: {len(val_loader.dataset)}")
        print(f"test: {len(test_loader.dataset)}")
    except FileNotFoundError as e:
        train_loader = val_loader = test_loader = None
        print("warning: dataloader creation failed. set DATA_DIR correctly") 