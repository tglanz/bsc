import torch
import math
import dataclasses

from typing import Sequence, Tuple

class Conv2d(torch.nn.Module):
    """ Our own implementation of 2d convolutional layer """
    def __init__(self,
                 in_channels: int = 1, out_channels: int = 1,
                 kernel_size: Tuple[int] = (1, 1),
                 stride: int = 1,
                 padding: int = 0):
        """
        Parameters:
          in_channels {int} Number of channels in the input tensor.

          out_channels {int} Number of channels in the output tensor.

          kernel_size {Tuple[int, int]} Kernel size in the form of (rows, cols).

          stride {int} The stride of the kernel indicates the step size by which the filter is moved along the input.
          This value indicates the step size in both the horizontal and vertical directions.

          padding {int} The amount of zeros to pad the input tensor before and after along each of it's spatial dimemsions.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = self.create_kernel()

    def create_kernel(self) -> torch.Tensor:
        """
        Create and initialize a kernel tensor.
        The tensor is initialized using a method similar to Xavier.
        The values are sampled from a distribution that is dependant on the size of the kernel.

        Returns a tensor of size (self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).
        Intuitively, we can think of this tensor as {self.out_channels} filters,
        each having {self.in_channels} channels and of size {self.kernel_size}.
        """
        xavier = 1 / math.sqrt(torch.Size(self.kernel_size).numel())
        kernel = torch.rand(self.out_channels, self.in_channels, *self.kernel_size)
        return kernel.uniform_(-xavier, +xavier)

    def convolve(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the 2d convolution of x with self.kernel.

        The kernel is comprised of multiple filters.

        Each filter is moved along the input according to {self.stride} and is reduced along the (c, h, w) dimensions.
        Each filter yields an output feature map, thus the result has {self.out_channels} channels. 
        """
        stride = self.stride
        p = self.padding
        xb, xc, xh, xw = x.shape

        # Create a padded tensor by first creating a zero tensor and then embed x in the center.
        padded = torch.zeros(xb, xc, xh + 2*p, xw + 2*p)
        padded[:, :, p:(p + xh), p:(p + xw)] = x

        y_size = self.calculate_output_size(x)
        y = torch.empty(y_size)

        kernel = self.weight
        kh, kw = self.kernel_size

        # Traverse the (row=r, col=s) in the output spatial dimensions
        # and compute all neurons of different channels simultaneously,
        for r in range(y.shape[2]):
            for s in range(y.shape[3]):
                sub_x = padded[:, :, r*stride:r*stride+kh, s*stride:(s*stride+kw)]

                # This unsqueeze is important.
                # sub_x has shape (batch, in_channels, kernel_h, kernel_w)
                # kernel has shape (out_channels, in_channels, kernel_h, kernel_w)

                sub_x.unsqueeze_(1)
                mul = sub_x * kernel
                scalar = mul.sum(dim=(2, 3, 4))
                y[:, :, r, s] = scalar

        return y

    def calculate_output_size(self, x: torch.Tensor) -> Sequence[int]:
        """
        Calculates the size of the result of x convolve with self.kernel, it takes into consideration the padding and the stride.

        The computation, along some dimension, can be intuitively though of as this way:

        Let x, k, p and s be the input size, kernel size padding and stride respectively.
        
        The output size is given by
        $$
            \\frac{ (x + p) - (k + s) }{ s }
        $$

        - We add by (x + p) because it is the initial padded input size.
        - Then we subtract by (k + s) becase this is where the filter cannot reach due to its size and stride
        - Finally we divide by s becase we downsample the input due to the stride.

        This size is the same for eery batch and output channel.
        """
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

def test_convolution_layer():
    @dataclasses.dataclass
    class TestCase:
        batch: int
        in_channels: int
        out_channels: int
        input_size: Tuple[int, int]
        kernel_size: Tuple[int, int]
        stride: int
        padding: int

        print_tensors: bool = False

    test_cases = [
        TestCase(1, 1, 1, (4, 4), (2, 2), 1, 0, print_tensors=True),
        TestCase(2, 1, 1, (4, 4), (2, 2), 1, 0, print_tensors=True),
        TestCase(1, 3, 1, (4, 4), (2, 2), 1, 0, print_tensors=True),
        TestCase(1, 1, 2, (4, 4), (2, 2), 1, 0, print_tensors=True),
        TestCase(1, 1, 1, (4, 4), (2, 2), 2, 0, print_tensors=True),
        TestCase(2, 2, 2, (4, 4), (2, 2), 1, 2, print_tensors=False),
        TestCase(1, 1, 1, (1, 1), (1, 1), 2, 1, print_tensors=True),
        TestCase(10, 3, 2, (16, 16), (3, 3), 3, 2, print_tensors=False),
    ]

    for idx, test_case in enumerate(test_cases):
        print(f"####### Test Case {idx} #######", "\n")
        print("Parameters: ", dataclasses.asdict(test_case), "\n")
        batch, in_channels, out_channels, input_size, kernel_size, stride, padding, print_tensors = dataclasses.astuple(test_case)

        our_conv2d = Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        torch_conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        torch_conv2d.eval()
        torch_conv2d.weight = torch.nn.Parameter(our_conv2d.weight)

        x = torch.ones((batch, in_channels, *input_size))

        our_y = our_conv2d.forward(x)
        torch_y = torch_conv2d.forward(x)

        if not print_tensors:
            print("Skipping tensor print", "\n")
        else:
            print("Input: ", x, "\n")
            print("Kernel: ", our_conv2d.weight, "\n")
            print("Our output: ", our_y, "\n")
            print("Torch output: ", torch_y, "\n")

        same = torch.equal(our_y, torch_y)
        distance = torch.norm(our_y - torch_y).item()
        assert same or distance < 1e-5


def main():
    test_convolution_layer()

if __name__ == '__main__':
    main()
