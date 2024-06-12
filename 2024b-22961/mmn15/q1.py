import torch
import logging
import math

from typing import Sequence

from torch.nn.modules import padding

class Conv2d(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 1, out_channels: int = 1,
                 kernel_size: Sequence[int] = (1, 1),
                 stride: int = 1,
                 padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = self.create_kernel()

    def create_kernel(self) -> torch.Tensor:
        xavier = 1 / math.sqrt(torch.Size(self.kernel_size).numel())
        kernel = torch.rand(self.out_channels, self.in_channels, *self.kernel_size)
        return kernel.uniform_(-xavier, +xavier)

    def convolve(self, x: torch.Tensor) -> torch.Tensor:
        p = self.padding
        xb, xc, xh, xw = x.shape
        padded = torch.zeros(xb, xc, xh + 2*p, xw + 2*p)

        padded[:, :, p:(p + xh), p:(p + xw)] = x

        y_size = self.calculate_output_size(x)
        y = torch.empty(y_size)

        kernel = self.weight
        kh, kw = self.kernel_size

        for r in range(y.shape[2]):
            for s in range(y.shape[3]):
                sub_x = padded[:, :, r:(r+kh), s:(s+kw)]
                mul = sub_x * kernel
                scalar = mul.sum(dim=(1, 2, 3))
                y[0, :, r, s] = scalar

        return y

    def calculate_output_size(self, x: torch.Tensor) -> Sequence[int]:
        batches = x.shape[0]
        xh, xw = x.shape[2], x.shape[3]

        kh, kw = self.kernel_size

        ph, pw = (2*self.padding for _ in range(2))
        sh, sw = self.stride, self.stride

        yh = (xh - kh + ph + sh) // sh
        yw = (xw - kw + pw + sw) // sw

        return (batches, self.out_channels, yh, yw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convolve(x)

def main():
    print("")
    print("======================================")

    batch = 1
    in_channels = 2
    out_channels = 3
    input_size = (4, 4)
    kernel_size = (2, 2)
    stride = 1
    padding = 1

    our_conv2d = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    torch_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
    torch_conv2d.eval()
    torch_conv2d.weight = torch.nn.Parameter(our_conv2d.weight)

    x = torch.ones((batch, in_channels, *input_size))
    our_y = our_conv2d.forward(x)
    torch_y = torch_conv2d.forward(x)

    print("our y: ", our_y.shape, our_y, sep="\n")
    print("")
    print("torch y: ", torch_y.shape, torch_y, sep="\n")


    same = torch.equal(our_y, torch_y)
    distance = torch.norm(our_y - torch_y).item()
    print("The tensors are equal: ", same)
    print("Distance: ", distance)

    assert same or distance < 1e-6

if __name__ == '__main__':
    main()
