import torch
import torchvision
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

TRAINED_MEAN = [0.485, 0.456, 0.406]
TRAINED_STD = [0.229, 0.224, 0.225]

def create_cifar_datasets():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.PILToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float),
        torchvision.transforms.Normalize(mean=TRAINED_MEAN, std=TRAINED_STD)
    ])

    train = torchvision.datasets.CIFAR10("datasets", train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10("datasets", train=False, download=True, transform=transform)

    return train, test

def plot_cifar_samples(cifar: torchvision.datasets.CIFAR10, samples_per_class: int = 3):

    plt.figure()

    class_counters = {
        cifar.class_to_idx[clazz]: 0
        for clazz
        in cifar.classes
    }

    fig, axes = plt.subplots(len(class_counters), samples_per_class, figsize=(12, 12))
    fig.tight_layout()

    trained_std_tensor = torch.tensor(TRAINED_STD).unsqueeze(1).unsqueeze(1)
    trained_mean_tensor = torch.tensor(TRAINED_MEAN).unsqueeze(1).unsqueeze(1)

    for x, y in DataLoader(cifar):
        y = y.item()
        class_counter = class_counters[y]
        if class_counter < 3:
            grid_index = y * samples_per_class + class_counter
            axes = plt.subplot(len(class_counters), samples_per_class, grid_index + 1)
            axes.tick_params(which="both", size=0, labelsize=0)
            axes.set_title(cifar.classes[y])

            x = x.squeeze(0)
            x = x * trained_std_tensor + trained_mean_tensor
            x = x.permute(1, 2, 0)
            plt.imshow(x.numpy())
            class_counters[y] += 1

            # exit early without iterating the entire data set
            if sum(class_counters.values()) == samples_per_class * len(class_counters):
                break

    plt.show()

def load_resnet() -> torchvision.models.ResNet:
    return torchvision.models.resnet18(pretrained=True)

def main():
    train_ds, test_ds = create_cifar_datasets()
    plot_cifar_samples(train_ds)

    input("Press any key")

if __name__ == '__main__':
    main()