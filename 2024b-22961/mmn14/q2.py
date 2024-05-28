import math
import torch
import torchvision
import torch.nn as nn
import torch.cuda

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from typing import Optional, Sequence, Tuple, TypeVar

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

def get_fashion_mnist_datasets():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float),
        torchvision.transforms.Lambda(torch.flatten)
    ])

    train_ds = torchvision.datasets.FashionMNIST("./datasets", train=True, download=True, transform=transform)
    test_ds = torchvision.datasets.FashionMNIST("./datasets", train=False, download=True, transform=transform)

    return train_ds, test_ds

class ModelA(nn.Module):
    def __init__(self, in_features: int = 784, out_features: int = 10):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.BatchNorm1d(in_features // 2),
            nn.Dropout(),
            nn.Linear(in_features=in_features // 2, out_features=out_features),
            nn.BatchNorm1d(out_features),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    
class ModelB(nn.Module):
    def __init__(self, in_features: int = 784, out_features: int = 10):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.BatchNorm1d(in_features // 2),
            nn.Dropout(),
            nn.Linear(in_features=in_features // 2, out_features=out_features),
            nn.BatchNorm1d(out_features),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

def train_model(model: nn.Module, train_dl: DataLoader, epochs: int, batch_limit: Optional[int] = None) -> Sequence[float]:
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    accuracy_per_epoch = [0] * epochs

    batches = len(train_dl)

    model.train(True)
    for epoch in range(epochs):
        for batch, (xs, ys) in enumerate(train_dl):

            if batch_limit is not None and batch > batch_limit:
                break

            optimizer.zero_grad()
            ys_model = model.forward(xs)
            loss = loss_fn(ys_model, ys)
            loss.backward()

            # track accuracies
            inferred_classes = torch.argmax(ys_model, dim=1)
            accuracy = (inferred_classes == ys).sum() / len(ys)
            accuracy_per_epoch[epoch] += accuracy.item()

            if batch % 100 == 0:
                print(f">  Epoch={epoch}, Batch={batch}/{batches}, Accuracy={accuracy * 100:.2f}%")

            optimizer.step()
        
        accuracy_per_epoch[epoch] /= len(train_dl)
        scheduler.step(accuracy_per_epoch[epoch])
        print(f"Epoch {epoch} has a mean accuracy of {accuracy_per_epoch[epoch] * 100:.2f}%")

    return accuracy_per_epoch

def eval_model(model: nn.Module, test_dl: DataLoader) -> Tuple[float, float]:
    batches = len(test_dl)
    accuracy_per_batch = [0] * batches

    model.eval()
    for batch, (xs, ys) in enumerate(test_dl):
        ys_model = model.forward(xs)

        inferred_classes = torch.argmax(ys_model, dim=1)
        accuracy = (inferred_classes == ys).sum() / len(ys)
        accuracy_per_batch[batch] = accuracy

    sigma2, mu = torch.var_mean(torch.tensor(accuracy_per_batch))
    return sigma2, mu

def normal_density_function(tensor: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    a = sigma.mul(math.sqrt(2 * math.pi))
    b = torch.pow(((tensor - mu)/sigma), 2).mul(-1/2)
    return tensor.div(a).mul(torch.exp(b))

def plot_model_stats(
        model_name: str,
        eval_normal_distribution: Tuple[torch.Tensor, torch.Tensor],
        train_accuracy_per_epoch: Sequence[float]):
    sigma2, mu = eval_normal_distribution
    
    fig = plt.figure()
    fig.suptitle(f"Model Stats: {model_name}. Mean accuracy {mu * 100:.0f}%")

    ax = plt.subplot(1, 2, 1)
    plt.plot([acc * 100 for acc in train_accuracy_per_epoch])
    ax = fig.axes[0]
    ax.set_title("Training accuracies")
    ax.set_xlabel("Epoch")
    xticks = list(range(len(train_accuracy_per_epoch)))
    ax.set_ylabel("Accuracy (%)")

    ax = plt.subplot(1, 2, 2)
    sigma = sigma2.sqrt()
    # bell_x = torch.linspace(mu - 6 * sigma, mu + 6 * sigma, 100)
    bell_x = torch.linspace(0, 1, 100)
    bell_y = normal_density_function(bell_x, sigma, mu)
    ax.set_title(f"Batch accuracy distribution")
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Density")
    plt.vlines([mu - 3 * sigma, mu + 3 * sigma], bell_y.min(), bell_y.max(), colors="r", linestyles="dotted")
    plt.plot(bell_x.numpy(), bell_y.numpy())

    fig.show()

def main():
    # shape = (2, 4, 3)
    # x = torch.arange(24).reshape(shape).float()
    # layer = DropNorm(torch.Size(shape[1:]))
    # layer.train(True)
    # layer.forward(x)

    train_ds, test_ds = get_fashion_mnist_datasets()
    train_dl = DataLoader(train_ds, batch_size=256)
    test_dl = DataLoader(test_ds, batch_size=256)

    in_features = next(iter(train_dl))[0][0].shape[0]
    out_features = len(train_ds.classes)

    model_a = "Model A", nn.Sequential(
        nn.Linear(in_features=in_features, out_features=in_features // 2),
        nn.BatchNorm1d(in_features // 2),
        nn.Dropout(),
        nn.Linear(in_features=in_features // 2, out_features=out_features),
        nn.BatchNorm1d(out_features),
        nn.Dropout(),
        nn.LogSoftmax(dim=1),
    )

    model_b = "Model B", nn.Sequential(
        nn.Linear(in_features=in_features, out_features=in_features // 2),
        DropNorm(in_features=torch.Size([in_features])),
        nn.Linear(in_features=in_features // 2, out_features=out_features),
        DropNorm(in_features=torch.Size([in_features])),
        nn.LogSoftmax(dim=1),
    )

    for name, model in (model_a, model_b):
        train_accuracy_per_epoch = train_model(model, train_dl, 5, batch_limit=1)
        eval_normal_distribution = eval_model(model, test_dl)
        print(f"{name} has reached mean accuracy of {eval_normal_distribution[1] * 100:.0f}%")
        plot_model_stats(name, eval_normal_distribution, train_accuracy_per_epoch)

    input("Press any key to continue")


if __name__ == '__main__':
    main()