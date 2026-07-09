"""SmallCNN — the clean, reusable CNN assembled in the CNN walkthrough (nb/cnn).

This is the exact model built and trained in `01_whole_game`, now extracted as an importable module:
a `features` conv trunk (`conv -> relu`, downsampling 28->14->7 while channels grow 1->16->32->64)
then a `flatten -> Linear` head to 10 class scores. Every design choice here is *justified, measured*
in walkthroughs 02-06:

  - the Conv2d op + output-size pyramid (28->14->7) ......... 02_features
  - conv over a flatten+MLP (locality, equivariance, params)  03_why_conv
  - the `conv -> relu -> conv` stacking (receptive field/ReLU) 04_stack_relu
  - stride-2 downsampling + growing channels ............... 05_downsample
  - the `flatten -> Linear` head + cross-entropy ........... 06_head_loss

The module layout (features.{0,2,4}, head.1) matches 01 exactly, so 01's saved weights
(`checkpoints/smallcnn.pt`) load straight into it — see 07_clean_model.
"""
from __future__ import annotations

import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),             # 28x28, local patterns
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 28->14, channels 16->32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 14->7,  channels 32->64
        )
        self.head = nn.Sequential(
            nn.Flatten(),                                                      # (B,64,7,7) -> (B,3136)
            nn.Linear(64 * 7 * 7, n_classes),                                  # -> (B,10) class scores
        )

    def forward(self, x):
        return self.head(self.features(x))
