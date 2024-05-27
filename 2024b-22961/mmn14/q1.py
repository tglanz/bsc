import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import math

class SplitLayer(nn.Module):
    input_batches: int
    input_features: int
    split_size: int
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
        """ Weight Initialization using Xavier's method """
        # sqrt(6 / (split_size + split_size))
        xavier = math.sqrt(3 / self.split_size) 

        with torch.no_grad():
            self.hidden[0].weight.uniform_(-xavier, +xavier)

        # Same as: (in fact I saw how to manually do it there)
        # torch.nn.init.uniform_(self.hidden[0].weight, -xavier, +xavier)

    def forward(self, x: torch.Tensor, verbose=False):

        log = print if verbose else lambda x: ()

        log(f"Input: {x}")

        a_in, b_in = x.split(split_size=self.split_size, dim=self.split_dim)
        log(f"Split A: {a_in}")
        log(f"Split B: {b_in}")


        a_out = self.hidden(a_in)
        log(f"Ouput A: {a_in}")

        b_out = self.hidden(b_in)
        log(f"Output B: {b_in}")

        y = torch.concat((a_out, b_out), dim=self.split_dim)
        log(f"Output: {y}")

        return y
    
    @staticmethod
    def demonstrate():
        x = torch.arange(12).reshape((3, 4)).float()

        split_layer = SplitLayer(3, 4)
        split_layer.train(False)
        
        split_layer.forward(x, verbose=True)


def main():
    SplitLayer.demonstrate()

if __name__ == '__main__':
    main()