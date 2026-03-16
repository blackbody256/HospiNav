import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOModel(nn.Module):
    def __init__(self, input_shape, grid_size, num_classes):
        super(YOLOModel, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        
        # Simple Backbone (Custom CNN)
        # Input: 3 x 224 x 224
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 28x28
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 14x14
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 7x7
        )
        
        # Fully Convolutional Head
        self.head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 5 + num_classes, kernel_size=1),
            nn.Sigmoid() # Output: (5 + num_classes) x 7 x 7
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

def yolo_loss(y_pred, y_true):
    """
    Improved YOLO loss in PyTorch.
    y_true/y_pred shape: (batch, 5 + num_classes, grid, grid)
    """
    obj_mask = y_true[:, 0:1, :, :] # 1 if object exists, (batch, 1, G, G)
    no_obj_mask = 1 - obj_mask
    
    # Objectness Loss
    # We use BCE for objectness
    obj_loss = F.binary_cross_entropy(y_pred[:, 0:1, :, :], y_true[:, 0:1, :, :], reduction='none')
    obj_loss = (obj_mask * obj_loss).mean()
    
    no_obj_loss = F.binary_cross_entropy(y_pred[:, 0:1, :, :], y_true[:, 0:1, :, :], reduction='none')
    no_obj_loss = (no_obj_mask * no_obj_loss).mean()
    
    # Bounding Box Loss (MSE for normalized coordinates)
    bbox_loss = F.mse_loss(obj_mask * y_pred[:, 1:5, :, :], obj_mask * y_true[:, 1:5, :, :], reduction='mean')
    
    # Classification Loss (BCE)
    class_loss = F.binary_cross_entropy(y_pred[:, 5:, :, :], y_true[:, 5:, :, :], reduction='none')
    class_loss = (obj_mask * class_loss).mean()
    
    # Weighting factors
    total_loss = 5.0 * obj_loss + 1.0 * no_obj_loss + 10.0 * bbox_loss + 1.0 * class_loss
    return total_loss

if __name__ == "__main__":
    model = YOLOModel((224, 224, 3), 7, 8)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)
