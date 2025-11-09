import random
import numpy as np

import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv3Classifier(nn.Module):
    def __init__(self, pretrained_name: str, num_classes: int, freeze_backbone: bool = True, unfreeze_last_n: int = 0):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

        # Handle freezing/unfreezing of backbone layers
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if unfreeze_last_n > 0 and hasattr(self.backbone, "layer"):
            for layer in self.backbone.layer[-unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        print(f"Model initialized. Backbone frozen: {freeze_backbone}, last {unfreeze_last_n} layers unfrozen.")
        print(f"Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, pixel_values):
        if all(not p.requires_grad for p in self.backbone.parameters()):
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values)
        else:
            outputs = self.backbone(pixel_values=pixel_values)

        feats = getattr(outputs, "pooler_output", None)
        if feats is None or feats.ndim != 2:
            feats = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(feats)
        return logits

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
