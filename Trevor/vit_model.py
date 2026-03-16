import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchExtract(nn.Module):
    def __init__(self, patch_size):
        super(PatchExtract, self).__init__()
        self.patch_size = patch_size
        # Using Conv2d to extract patches: kernel_size=stride=patch_size
        # This will output (B, projection_dim, H/patch, W/patch)
        # But we'll do it manually to match the previous logic if needed, 
        # or just use Conv2d which is standard for ViT.
        # Let's use Conv2d for embedding later, here just extraction.
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W)
        patches = self.unfold(x) # (B, 3 * patch*patch, num_patches)
        patches = patches.transpose(1, 2) # (B, num_patches, patch_dims)
        return patches

class PatchEmbedding(nn.Module):
    def __init__(self, num_patches, patch_dims, projection_dim):
        super(PatchEmbedding, self).__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(patch_dims, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, patch):
        # patch: (B, num_patches, patch_dims)
        positions = torch.arange(0, self.num_patches, step=1, device=patch.device)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class TransformerBlock(nn.Module):
    def __init__(self, projection_dim, num_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(projection_dim)
        self.attn = nn.MultiheadAttention(projection_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(projection_dim)
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, projection_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim * 2, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, num_patches, projection_dim)
        x1 = self.norm1(x)
        attn_output, _ = self.attn(x1, x1, x1)
        x2 = x + attn_output
        x3 = self.norm2(x2)
        x3 = self.mlp(x3)
        return x2 + x3

class ViTDetector(nn.Module):
    def __init__(self, input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_layers, grid_size, num_classes):
        super(ViTDetector, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        patch_dims = 3 * patch_size * patch_size
        self.patch_extract = PatchExtract(patch_size)
        self.patch_embed = PatchEmbedding(num_patches, patch_dims, projection_dim)
        
        self.transformers = nn.Sequential(*[
            TransformerBlock(projection_dim, num_heads) for _ in range(transformer_layers)
        ])
        
        # Reshape and Head
        # Assuming num_patches is perfect square
        self.h_patches = int(num_patches**0.5)
        self.w_patches = int(num_patches**0.5)
        
        self.conv_head = nn.Sequential(
            nn.Conv2d(projection_dim, projection_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(projection_dim * 2, 5 + num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, 3, 224, 224)
        patches = self.patch_extract(x) # (B, num_patches, patch_dims)
        x = self.patch_embed(patches) # (B, num_patches, projection_dim)
        x = self.transformers(x) # (B, num_patches, projection_dim)
        
        # Reshape to (B, projection_dim, h, w)
        x = x.transpose(1, 2).reshape(-1, x.size(-1), self.h_patches, self.w_patches)
        
        # Pool to grid size if necessary
        if self.h_patches != self.grid_size:
            x = F.interpolate(x, size=(self.grid_size, self.grid_size), mode='bilinear', align_corners=False)
            
        x = self.conv_head(x)
        return x

if __name__ == "__main__":
    img_size = 224
    patch_size = 16
    num_patches = (img_size // patch_size) ** 2
    
    model = ViTDetector(
        input_shape=(3, img_size, img_size),
        patch_size=patch_size,
        num_patches=num_patches,
        projection_dim=64,
        num_heads=4,
        transformer_layers=4,
        grid_size=7,
        num_classes=8
    )
    dummy_input = torch.randn(1, 3, img_size, img_size)
    output = model(dummy_input)
    print("Output shape:", output.shape)
