import torch
import torch.nn as nn
import torch.cuda
from typing import Optional, Sequence, TypeVar

T = TypeVar('T')

class DropNorm(nn.Module):

    def __init__(self,
                 in_features: torch.Size,
                 device: Optional[torch.device] = None,
                 epsilon: float = 10e-5,
                 mu_avg_coeff: float = 0.9,
                 sigma2_avg_coeff: float = 0.9):
        """

        Parameters
          input_shape {torch.Size} indicates the shape of the inputs. The first dimension is the batch size.
          epsilon {float} is used to avoid zero division.
            According to the chapter on Normalization layers (bullet 1), it is usually 10e-5.
        """
        super().__init__()

        assert in_features.numel() % 2 == 0, "Expected number of elements to be even"

        self.epsilon = epsilon
        self.in_features = in_features

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
        self.device = device

        # According to the chapter on Normalization layers (bullet 2), the input might not be best kept normalized.
        # So, we want to learn the parameters (beta and gamma) to translate and scale them 
        self.gamma = nn.Parameter(torch.ones(in_features, device=device))
        self.beta = nn.Parameter(torch.zeros(in_features, device=device))

        # As taught, it is beneficial to keep track of the averages of the mean and variances to use in inference time. 
        self.mu_avg_coeff = mu_avg_coeff
        self.mu_avg = torch.tensor(0, device=device)
        self.sigma2_avg_coeff = sigma2_avg_coeff
        self.sigma2_avg = torch.tensor(1, device=device)

    def create_features_mask(self) -> torch.Tensor:
        zeros_count = self.in_features.numel() // 2
        ones_count = zeros_count
        mask = torch.concat((
            torch.zeros(zeros_count, device=self.device),
            torch.ones(ones_count, device=self.device)))
        mask = mask[torch.randperm(mask.numel())]
        mask = mask.reshape(self.in_features)
        return mask == 1

    def forward(self, x: torch.Tensor):
        mask = self.create_features_mask()

        if self.training:
            sub_x = x[:, mask]
            sigma2, mu = torch.var_mean(sub_x, dim=1)

            with torch.no_grad():
                self.mu_avg = self.mu_avg_coeff * self.mu_avg + (1 - self.mu_avg_coeff) * mu
                self.sigma2_avg = self.sigma2_avg_coeff * self.sigma2_avg + (1 - self.sigma2_avg_coeff) * sigma2
        else:
            sigma2 = self.sigma2_avg
            mu = self.mu_avg

        # clone so we don't mutate input tensor and set masked features according to normalization
        x_hat = torch.zeros(x.shape, device=self.device)
        x_hat[:, mask] = (sub_x.flatten(1) - mu.unsqueeze(1)) / torch.sqrt(sigma2.unsqueeze(1)**2 + self.epsilon)
        y = self.gamma * x_hat + self.beta

        return y

def main():
    shape = (2, 4, 3)
    x = torch.arange(24).reshape(shape).float()
    layer = DropNorm(torch.Size(shape[1:]))
    layer.train(True)
    layer.forward(x)

if __name__ == '__main__':
    main()