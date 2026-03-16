import torch
import torch.nn as nn
import torch.nn.functional as F

class SSDModel(nn.Module):
    def __init__(self, num_classes):
        super(SSDModel, self).__init__()
        self.num_classes = num_classes 
        self.num_classes_with_bg = num_classes + 1
        # Backbone produces 56x56, 28x28, 14x14 from 224x224 input
        self.feature_maps = [56, 28, 14]
        self.n_anchors = 4
        self.ar = [1, 2, 0.5, 1.5]

        # Backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) 
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        
        # Heads
        self.loc_56 = nn.Conv2d(128, self.n_anchors * 4, kernel_size=3, padding=1)
        self.loc_28 = nn.Conv2d(256, self.n_anchors * 4, kernel_size=3, padding=1)
        self.loc_14 = nn.Conv2d(512, self.n_anchors * 4, kernel_size=3, padding=1)
        
        self.conf_56 = nn.Conv2d(128, self.n_anchors * self.num_classes_with_bg, kernel_size=3, padding=1)
        self.conf_28 = nn.Conv2d(256, self.n_anchors * self.num_classes_with_bg, kernel_size=3, padding=1)
        self.conf_14 = nn.Conv2d(512, self.n_anchors * self.num_classes_with_bg, kernel_size=3, padding=1)

        # Generate anchors
        self.anchors = self._generate_anchors()

    def _generate_anchors(self):
        anchors = []
        scales = [0.07, 0.15, 0.3] # Adjusted for higher resolutions
        
        for i, f_size in enumerate(self.feature_maps):
            for y in range(f_size):
                for x in range(f_size):
                    cx = (x + 0.5) / f_size
                    cy = (y + 0.5) / f_size
                    s = scales[i]
                    for ar in self.ar:
                        w = s * (ar**0.5)
                        h = s / (ar**0.5)
                        anchors.append([cx, cy, w, h])
        return torch.tensor(anchors, dtype=torch.float32)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = self.pool(x) # 112
        x = F.relu(self.conv2(x))
        x = self.pool(x) # 56
        
        f56 = F.relu(self.conv3(x)) # 56x56
        x = self.pool(f56) # 28
        
        f28 = F.relu(self.conv4(x)) # 28x28
        x = self.pool(f28) # 14
        
        f14 = F.relu(self.conv5(x)) # 14x14
        
        loc56 = self.loc_56(f56).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        loc28 = self.loc_28(f28).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        loc14 = self.loc_14(f14).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        
        conf56 = self.conf_56(f56).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes_with_bg)
        conf28 = self.conf_28(f28).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes_with_bg)
        conf14 = self.conf_14(f14).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes_with_bg)
        
        loc_preds = torch.cat([loc56, loc28, loc14], 1)
        conf_preds = torch.cat([conf56, conf28, conf14], 1)
        
        return loc_preds, conf_preds

def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    a = torch.cat([box_a[:, :2] - box_a[:, 2:]/2, box_a[:, :2] + box_a[:, 2:]/2], 1)
    b = torch.cat([box_b[:, :2] - box_b[:, 2:]/2, box_b[:, :2] + box_b[:, 2:]/2], 1)
    inter = intersect(a, b)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union

def match(threshold, truths, anchors, num_classes):
    if truths.size(0) == 0:
        conf = torch.zeros(anchors.size(0), dtype=torch.long, device=truths.device)
        loc = torch.zeros(anchors.size(0), 4, device=truths.device)
        return loc, conf
    overlaps = jaccard(truths[:, 1:], anchors)
    best_anchor_overlap, best_anchor_idx = overlaps.max(1)
    best_truth_overlap, best_truth_idx = overlaps.max(0)
    for j in range(best_anchor_idx.size(0)):
        best_truth_idx[best_anchor_idx[j]] = j
        best_truth_overlap[best_anchor_idx[j]] = 2
    conf = truths[best_truth_idx, 0].long() + 1
    conf[best_truth_overlap < threshold] = 0
    loc = truths[best_truth_idx, 1:]
    return loc, conf

def multibox_loss(loc_preds, conf_preds, loc_targets, conf_targets):
    pos = conf_targets > 0
    num_pos = pos.sum(dim=1, keepdim=True)
    pos_idx = pos.unsqueeze(dim=-1).expand_as(loc_preds)
    loc_p = loc_preds[pos_idx].view(-1, 4)
    loc_t = loc_targets[pos_idx].view(-1, 4)
    loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')
    batch_conf = conf_preds.view(-1, conf_preds.size(-1))
    loss_c = F.cross_entropy(batch_conf, conf_targets.view(-1), reduction='none')
    loss_c = loss_c.view(conf_preds.size(0), -1)
    loss_c_pos = loss_c[pos]
    loss_c_neg = loss_c.clone()
    loss_c_neg[pos] = 0
    _, loss_idx = loss_c_neg.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)
    num_neg = torch.clamp(3 * num_pos, max=pos.size(1) - 1)
    neg = idx_rank < num_neg
    loss_c_final = loss_c[pos | neg].sum()
    N = num_pos.sum().float()
    if N > 0:
        return (loss_l + loss_c_final) / N
    else:
        return loss_c_final.sum()

if __name__ == "__main__":
    num_classes = 8
    model = SSDModel(num_classes)
    dummy_input = torch.randn(1, 3, 224, 224)
    loc, conf = model(dummy_input)
    print("Loc shape:", loc.shape) # Total: (56*56 + 28*28 + 14*14) * 4 = (3136 + 784 + 196) * 4 = 4116 * 4 = 16464
    print("Conf shape:", conf.shape)
    print("Anchors shape:", model.anchors.shape)
