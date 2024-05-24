import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

class SplitLayer(nn.Module):

    input_batches: int
    input_features: int
    split_size: int
    split_dim: int

    def __init__(self, input_batches: int, input_features: int):
        super(SplitLayer, self).__init__()

        assert input_batches > 0, "expected input_batches to be positive"
        assert input_features % 2 == 0, "expected input_features to be even"

        self.input_batches = input_batches
        self.input_features = input_features

        self.split_dim = 1
        self.split_size = self.input_features // 2

        self.hidden = nn.Sequential(
            nn.Linear(self.split_size, self.split_size),
            nn.ReLU())

    def forward(self, x: torch.Tensor):
        a_in, b_in = x.split(split_size=self.split_size, dim=self.split_dim)

        print("x shape", x.shape)
        print("a shape", a_in.shape)
        a_out = self.hidden(a_in)
        b_out = self.hidden(b_in)

        return torch.concat((a_out, b_out), dim=self.split_dim)

def main():
    x = torch.arange(12).reshape((3, 4)).float()
    split_layer = SplitLayer(3, 4)
    y = split_layer.forward(x)
    print(y)

if __name__ == '__main__':
    main()