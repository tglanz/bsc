from typing import List, Optional, Sequence, Set
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from matplotlib import pyplot as plt

df = pd.read_csv("assets/diabetes.csv", sep="\t")
df['Class'] = pd.qcut(df.Y, 10, labels=False)

def visualize_deciles():
    fig = plt.figure()
    plt.scatter(df.Class, df.Y)
    ax = fig.axes[0]
    ax.set_xlabel("Class")
    ax.set_ylabel("Y")
    ax.set_title("Diabetes Y Classes")
    ax.set_xticks(range(10), (str(i+1) for i in range(10)))

# Our custom Dataset. Useful reference:
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class DiabetesDataset(torch_data.Dataset):
    df: pd.DataFrame

    x_include: Set[str]
    y_column: str

    def __init__(self,
                 df: pd.DataFrame,
                 x_include: Sequence[str],
                 y_column: str
    ):
        self.df = df

        self.x_include = set(x_include)
        self.y_column = y_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index) -> torch.Tensor:
        x = self.df.loc[index, (col in self.x_include for col in df.columns)].values
        y = self.df.loc[index, df.columns == self.y_column].values
        return (
            torch.tensor(x).float(),
            torch.tensor(y).squeeze().long())

dataset_with_y = DiabetesDataset(
    df,
    x_include=[col for col in df.columns if col not in ("Class")],
    y_column="Class")

dataloader_with_y = torch_data.DataLoader(dataset_with_y, 10)

dataset_without_y = DiabetesDataset(
    df,
    x_include=[col for col in df.columns if col not in ("Class", "Y")],
    y_column="Class")

dataloader_without_y = torch_data.DataLoader(dataset_without_y, 10)

# print the first batch
print(next(iter(dataloader_with_y)))

class ClassPredictor:
    dataloader: torch_data.DataLoader

    def __init__(self,
                 dataloader: torch_data.DataLoader,
                 in_features: int,
                 out_features: int = 10,
                 learning_rate: float = 0.1,
    ):
        self.dataloader = dataloader

        self.model = nn.Sequential(
            nn.Linear(in_features, in_features * 2),
            nn.ReLU(),
            nn.Linear(in_features * 2, out_features),
            nn.LogSoftmax(dim=1)
        )

        self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate)

    def forward_batch(self, batch):
        xs, ys = batch

        # new batch, zero all gradients
        self.optimizer.zero_grad()

        # feed forwad
        output = self.model(xs)

        # compute loss and gradients
        loss = self.loss(output, ys)
        loss.backward()

        # update model parameters
        self.optimizer.step()

        # calculate the accuracy.
        # `inferred_classes` gets, from each entry in the batch the highest probability class.
        # then, we count how many classes the model got right, and divide by total classes
        # in order to normalize for a relative error.
        inferred_classes = output.argmax(dim=1)
        accuracy = (inferred_classes == ys).sum() / len(ys)

        return loss.detach(), accuracy.detach()

    def forward(self):
        total_batches = len(self.dataloader)
        loss_overtime = torch.empty(total_batches)
        accuracy_overtime = torch.empty(total_batches)

        for idx, batch in enumerate(self.dataloader):
            loss, accuracy = self.forward_batch(batch)
            loss_overtime[idx] = loss
            accuracy_overtime[idx] = accuracy

        return loss_overtime.mean(), accuracy_overtime.mean()

predictor = ClassPredictor(
    dataloader=dataloader_with_y,
    in_features=len(df.columns) - 1,
    learning_rate=0.01)

epochs = 50
losses = torch.empty(epochs)
accuracies = torch.empty(epochs)

for epoch in range(epochs):
    print(f"Training, epoch={epoch}")
    loss, accuracy = predictor.forward()
    losses[epoch] = loss
    accuracies[epoch] = accuracy

fig = plt.figure()
plt.subplot(1, 2, 1)
plt.title("Loss")
plt.plot(range(epochs), losses)
plt.xlabel("Epoch")

plt.subplot(1, 2, 2)
plt.plot(range(epochs), accuracies)
plt.title("Accuracy")
plt.xlabel("Epoch")

fig.show()
input()