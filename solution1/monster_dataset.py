from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class CanDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the images (can be train or test).
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool, optional): Whether the dataset is for test set (no labels).
        """
        self.root_dir = root_dir
        self.transform = transform
        #self.is_test = is_test  # Flag to indicate if it's the test dataset
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(label)  # Add label only for training set
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
