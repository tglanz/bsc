import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import math
class SplitLayer(nn.Module):
    """ 
    An implementation of a neural network layer that given an input X it does the following:
    - Split X into two halves X1, X2
    - Feed X1, X2 into the same LinearLayer yielding outputs Z1, Z2
    - Feed Z1, Z2 into a ReLU layer yielding outputs Y1, Y2
    - Concatenate Y=[Y1, Y2]
    """

    # The number of input batches
    input_batches: int

    # The number of input features.
    # Because we split the input to two halves, this is expected to be even.
    input_features: int

    # Determines the size of each split.
    # It is basically input_features/2
    split_size: int

    # Determines the dimension to split by.
    # It is basically 1 since 1 is the expected features dimension.
    split_dim: int

    def __init__(self, input_batches: int, input_features: int):
        super().__init__()

        assert input_batches > 0, "expected input_batches to be positive"
        assert input_features % 2 == 0, "expected input_features to be even"

        self.input_batches = input_batches
        self.input_features = input_features

        self.split_dim = 1
        self.split_size = self.input_features // 2

        self.hidden = nn.Sequential(
            nn.Linear(self.split_size, self.split_size),
            nn.ReLU())

        self.initialize_weights()

    def initialize_weights(self):
        """
        Weight Initialization using Xavier's method.

        As explained here
        - https://d2l.ai/chapter_multilayer-perceptrons/numerical-stability-and-init.html#xavier-initialization
        
        Xavier's method tackles the issue of exploding/vanishing gradients 
        """
        # sqrt(6 / (split_size + split_size))
        xavier = math.sqrt(3 / self.split_size) 

        with torch.no_grad():
            self.hidden[0].weight.uniform_(-xavier, +xavier)

        # Same as: (in fact I saw how to manually do it there)
        # torch.nn.init.uniform_(self.hidden[0].weight, -xavier, +xavier)

        # Bias is kept as is (0 bias)

    def forward(self, x: torch.Tensor, verbose=False):
        log = print if verbose else lambda x: ()

        log(f"  Input: {x}")

        a_in, b_in = x.split(split_size=self.split_size, dim=self.split_dim)
        log(f"  Split A: {a_in}")
        log(f"  Split B: {b_in}")

        a_out = self.hidden(a_in)
        log(f"  Ouput A: {a_in}")

        b_out = self.hidden(b_in)
        log(f"  Output B: {b_in}")

        y = torch.concat((a_out, b_out), dim=self.split_dim)
        log(f"  Output: {y}")

        return y

def demonstrate_split_layer():
    print("Demonstration of SplitLayer: ")
    
    x = torch.arange(12).reshape((3, 4)).float()
    
    split_layer = SplitLayer(3, 4)
    split_layer.train(False)
    
    split_layer.forward(x, verbose=True)

demonstrate_split_layer()