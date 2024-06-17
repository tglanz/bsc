from typing import Optional
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
        torchvision.transforms.Resize(224),
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

def replace_and_train_head(model: torch.nn.Module, data: DataLoader, class_count: int = 10, batch_limit: Optional[int] = None, device: torch.DeviceObjType = None):
    device = device or torch.device("cpu")

    # Turn off grad tracking of all paramaeters
    for parameter in model.parameters(recurse=True):
        parameter.requires_grad = False

    # Replace the fully connected layer with a new one to have the correct number of classes
    original_fc = model.fc
    model.fc = torch.nn.Linear(
        original_fc.in_features, class_count,
        bias=original_fc.bias is not None,
        device=original_fc.weight.device,
        dtype=original_fc.weight.dtype)
    

    batches = len(data)
    epochs = 2
    accuracy_per_epoch = [0]*epochs

    # Create the optimizer.
    # Notice how we only provide the parameters of the newly created layer
    optimizer = torch.optim.Adam(params=model.fc.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(params=model.fc.parameters(), lr=0.1, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)

    # Loss function well suited for classification tasks
    loss_fn = torch.nn.CrossEntropyLoss()

    print(device)
    model.to(device)
    model.train(True)

    for epoch in range(epochs):
        for batch_index, batch in enumerate(data):
            if batch_limit and batch_index >= batch_limit:
                break

            xs, ys = batch
            xs = xs.to(device)
            ys = ys.to(device)

            optimizer.zero_grad()
            ys_model = model(xs)
            loss = loss_fn(ys_model, ys)
            loss.backward()
            optimizer.step()

            # Track the accuracies
            inferred_classes = torch.argmax(ys_model, dim=1)
            accuracy = (inferred_classes == ys).sum() / len(ys)
            accuracy_per_epoch[epoch] += accuracy.item()

            print(f"FINISHED_BATCH: epoch={epoch}, batch={batch_index}/{batches}, accuracy={accuracy}")

        accuracy_per_epoch[epoch] /= batches
        print(f"FINISHED_EPOCH: epoch={epoch}, mean_accuracy={accuracy_per_epoch[epoch]*100:.2f}%")
        # scheduler.step(epoch)


def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)

    train_ds, test_ds = create_cifar_datasets()
    # plot_cifar_samples(train_ds)

    resnet = load_resnet()
    replace_and_train_head(
        resnet,
        DataLoader(train_ds, shuffle=True, batch_size=256, num_workers=4),
        device=device)
    
main()