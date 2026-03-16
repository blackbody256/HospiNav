import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from data_loader import load_data_config, create_pytorch_dataloader
from ssd_model import SSDModel, multibox_loss, match

# Constants
IMG_SIZE = 224
GRID_SIZE = 7
BATCH_SIZE = 16
EPOCHS = 10
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "./Hospital.v1-hospitaldata.yolov8"
)

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for images, _, raw_boxes, num_objs in dataloader:
        images = images.to(device)
        optimizer.zero_grad()
        
        loc_preds, conf_preds = model(images)
        # Match ground truth to anchors for each sample in batch
        batch_loc_t = []
        batch_conf_t = []
        for i in range(images.size(0)):
            if num_objs[i] > 0:
                truths = raw_boxes[i, :num_objs[i]].to(device)
                loc_t, conf_t = match(0.5, truths, model.anchors.to(device), model.num_classes)
            else:
                # No objects, all background
                loc_t = torch.zeros(model.anchors.size(0), 4).to(device)
                conf_t = torch.zeros(model.anchors.size(0), dtype=torch.long).to(device)
            batch_loc_t.append(loc_t)
            batch_conf_t.append(conf_t)
        
        loc_targets = torch.stack(batch_loc_t, 0)
        conf_targets = torch.stack(batch_conf_t, 0)
        loss = multibox_loss(loc_preds, conf_preds, loc_targets, conf_targets)
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, _, raw_boxes, num_objs in dataloader:
            images = images.to(device)
            loc_preds, conf_preds = model(images)
            batch_loc_t = []
            batch_conf_t = []
            for i in range(images.size(0)):
                if num_objs[i] > 0:
                    truths = raw_boxes[i, :num_objs[i]].to(device)
                    loc_t, conf_t = match(0.5, truths, model.anchors.to(device), model.num_classes)
                else:
                    loc_t = torch.zeros(model.anchors.size(0), 4).to(device)
                    conf_t = torch.zeros(model.anchors.size(0), dtype=torch.long).to(device)
                batch_loc_t.append(loc_t)
                batch_conf_t.append(conf_t)
            loc_targets = torch.stack(batch_loc_t, 0)
            conf_targets = torch.stack(batch_conf_t, 0)
            loss = multibox_loss(loc_preds, conf_preds, loc_targets, conf_targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    yaml_path = os.path.join(DATA_DIR, 'data.yaml')
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found.")
        return

    config = load_data_config(yaml_path)
    num_classes = config['nc']
    
    print(f"Loading dataset from {DATA_DIR}...")
    train_loader = create_pytorch_dataloader(DATA_DIR, 'train', IMG_SIZE, GRID_SIZE, num_classes, BATCH_SIZE, augment=True)
    val_loader = create_pytorch_dataloader(DATA_DIR, 'valid', IMG_SIZE, GRID_SIZE, num_classes, BATCH_SIZE)
    test_loader = create_pytorch_dataloader(DATA_DIR, 'test', IMG_SIZE, GRID_SIZE, num_classes, BATCH_SIZE)
    
    # Train SSD Model
    print("\n--- Training SSD Model ---")
    ssd_model = SSDModel(num_classes).to(device)
    optimizer_ssd = optim.Adam(ssd_model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(ssd_model, train_loader, optimizer_ssd, device)
        val_loss = evaluate(ssd_model, val_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('models', exist_ok=True)
            torch.save(ssd_model.state_dict(), 'models/ssd_hospital.pth')
            print("  (Model saved)")
    
    print("\nEvaluating on Test Set:")
    test_loss = evaluate(ssd_model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
