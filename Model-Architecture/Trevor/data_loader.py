import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml

def load_data_config(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_label_file(label_path):
    """
    Parses a YOLO label file.
    Returns a list of [class_id, x_center, y_center, width, height]
    """
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                data = list(map(float, line.strip().split()))
                if len(data) == 5:
                    boxes.append(data)
    return np.array(boxes)

def load_image_and_labels(image_path, label_path, img_size, num_classes, augment=False):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load labels
    boxes = parse_label_file(label_path)
    
    if augment:
        # Simple Augmentation
        if np.random.rand() > 0.5:
            # Horizontal Flip
            img = cv2.flip(img, 1)
            if len(boxes) > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1] # Update x_center
                
        # Random Brightness/Contrast
        alpha = 1.0 + (np.random.rand() * 0.2 - 0.1) # Contrast
        beta = np.random.rand() * 20 - 10 # Brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Transpose to CHW for PyTorch (C, H, W)
    img = np.transpose(img, (2, 0, 1))
    
    return img, boxes

def boxes_to_yolo_target(boxes, grid_size, num_classes):
    """
    Converts list of boxes to a YOLO-style grid target.
    Target shape: (grid_size, grid_size, 5 + num_classes)
    5: [objectness, x, y, w, h]
    Note: In PyTorch, we often keep this as (grid_size, grid_size, 5 + num_classes) 
    or (5 + num_classes, grid_size, grid_size). We'll stick to (5+C, G, G) for consistency with Conv2D.
    """
    target = np.zeros((5 + num_classes, grid_size, grid_size), dtype=np.float32)
    if boxes is None or len(boxes) == 0:
        return target
        
    for box in boxes:
        class_id, x, y, w, h = box
        # Find which grid cell the center belongs to
        grid_x = int(x * grid_size)
        grid_y = int(y * grid_size)
        
        # Clip to grid boundaries
        grid_x = max(0, min(grid_x, grid_size - 1))
        grid_y = max(0, min(grid_y, grid_size - 1))
        
        # If multiple objects fall in the same cell, the last one wins (simple YOLO)
        if target[0, grid_y, grid_x] == 0:
            target[0, grid_y, grid_x] = 1.0 # Objectness
            target[1:5, grid_y, grid_x] = [x, y, w, h]
            target[5 + int(class_id), grid_y, grid_x] = 1.0 # One-hot class
            
    return target

class HospitalDataset(Dataset):
    def __init__(self, data_dir, split, img_size, grid_size, num_classes, augment=False, max_objects=20):
        self.img_dir = os.path.join(data_dir, split, 'images')
        self.label_dir = os.path.join(data_dir, split, 'labels')
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.augment = augment
        self.max_objects = max_objects

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_file)
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)
        
        img, boxes = load_image_and_labels(img_path, label_path, self.img_size, self.num_classes, augment=self.augment)
        if img is None:
            # Return empty sample if image loading fails
            img = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
            target = np.zeros((5 + self.num_classes, self.grid_size, self.grid_size), dtype=np.float32)
            raw_boxes = np.zeros((self.max_objects, 5), dtype=np.float32)
            num_objs = 0
        else:
            target = boxes_to_yolo_target(boxes, self.grid_size, self.num_classes)
            # Pad raw boxes to max_objects
            num_objs = min(len(boxes), self.max_objects)
            raw_boxes = np.zeros((self.max_objects, 5), dtype=np.float32)
            if num_objs > 0:
                raw_boxes[:num_objs] = boxes[:num_objs]
            
        return torch.from_numpy(img), torch.from_numpy(target), torch.from_numpy(raw_boxes), torch.tensor(num_objs)

def create_pytorch_dataloader(data_dir, split, img_size, grid_size, num_classes, batch_size=16, augment=False):
    dataset = HospitalDataset(data_dir, split, img_size, grid_size, num_classes, augment=augment)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'),
        num_workers=0 # Set to 0 for simplicity/compatibility in some environments
    )
    return dataloader

if __name__ == "__main__":
    # Test loading one sample
    data_path = 'Hospital.v1-hospitaldata.yolov8'
    if os.path.exists(data_path):
        config = load_data_config(os.path.join(data_path, 'data.yaml'))
        print("Classes:", config['names'])
        
        dataloader = create_pytorch_dataloader(data_path, 'train', 224, 7, config['nc'], 1, augment=True)
        for img, target in dataloader:
            print("Image shape:", img.shape)
            print("Target shape:", target.shape)
            # Find where objectness is 1
            obj_indices = torch.where(target[0, 0, :, :] == 1)
            print("Object locations in grid:", list(zip(obj_indices[0].tolist(), obj_indices[1].tolist())))
            break
    else:
        print(f"Data directory {data_path} not found.")
