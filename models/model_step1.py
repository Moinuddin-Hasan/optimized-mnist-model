"""
Target:
The goal for this first step is to create a foundational CNN that validates the entire training pipeline. The objectives are:
1.  Build a simple, working CNN architecture that is easy to understand.
2.  Ensure the training and testing loops run correctly, data is loaded, and loss decreases as expected.
3.  Establish a baseline performance. At this stage, I am NOT aiming for the final constraints (<8k params, 99.4% accuracy). Instead, the goal is to create a model that gets decent accuracy (>98.5%) and then analyze its weaknesses. This model uses a large fully-connected layer, which is expected to be the main bottleneck.

Result:
- Test Accuracy: 98.95% (Peak)
- Parameters: 71,274
- Epochs to reach peak: 9

Analysis:
The model successfully establishes a baseline and proves the training pipeline works. The results clearly highlight the problems that need to be solved:
1.  **Parameters**: At 71,274, the parameter count is nearly 9 times the target limit of 8,000. The model summary shows that the vast majority of these (over 64k) are in the first fully-connected layer (`fc1`). This is extremely inefficient and is the primary target for optimization.
2.  **Accuracy**: A peak accuracy of 98.95% is a strong start, but it falls short of the 99.4% target. The model also shows signs of overfitting; the training accuracy reaches 99.8% while the test accuracy stagnates. This indicates that regularization is needed.
This analysis confirms that the next logical step is to completely remove the inefficient fully-connected layers and introduce regularization techniques like Batch Normalization and Dropout.
"""

import torch.nn as nn
import torch.nn.functional as F

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 10

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=20, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU()
        ) # output_size = 8
        
        # This fully connected layer is the main source of parameters
        # After the conv layers, the image is 8x8 with 20 channels. 8*8*20 = 1280.
        self.fc1 = nn.Linear(1280, 50) # <<<<<<<<<<<<<<< THIS LINE IS CORRECTED (was 320)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.view(-1, 1280) # <<<<<<<<<<<<<<< THIS LINE IS CORRECTED (was 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)