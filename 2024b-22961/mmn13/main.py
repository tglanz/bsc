from typing import List, Optional, Sequence, Set
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

epochs = 50
plot = True
lr = 0.01
batch_size = 10

diabetes_df = pd.read_csv("assets/diabetes.csv", sep="\t")
diabetes_df['Class'] = pd.qcut(diabetes_df.Y, 10, labels=False)

def visualize_deciles():
    fig = plt.figure()
    plt.scatter(diabetes_df.Class, diabetes_df.Y)
    ax = fig.axes[0]
    ax.set_xlabel("Class")
    ax.set_ylabel("Y")
    ax.set_title("Diabetes Y Classes")
    ax.set_xticks(range(10), (str(i+1) for i in range(10)))

# Our custom Dataset. Useful reference:
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class DiabetesDataset(Dataset):
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
        x = self.df.loc[index, (col in self.x_include for col in self.df.columns)].values
        y = self.df.loc[index, self.df.columns == self.y_column].values
        return (
            torch.tensor(x).float(),
            torch.tensor(y).squeeze().long())

class ClassPredictor:

    def __init__(self,
                 in_features: int,
                 out_features: int = 10,
                 learning_rate: float = 0.1,
    ):
        num_stages = 4
        layers = []
        for i in range(1, num_stages):
            layers.append(nn.Linear(in_features * i, in_features * (i + 1)))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(
            *layers,
            nn.Linear(in_features * num_stages, out_features),
            nn.LogSoftmax(dim=1)
        )

        self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate)

    def forward_batch(self, batch, optimize: bool = True):
        xs, ys = batch

        if optimize:
            # new batch, zero all gradients
            self.optimizer.zero_grad()

        # feed forwad
        output = self.model(xs)

        # compute loss and gradients
        loss = self.loss(output, ys)
        loss.backward()

        if optimize:
            # update model parameters
            self.optimizer.step()

        # calculate the accuracy.
        # `inferred_classes` gets, from each entry in the batch the highest probability class.
        # then, we count how many classes the model got right, and divide by total classes
        # in order to normalize for a relative error.
        inferred_classes = output.argmax(dim=1)
        accuracy = (inferred_classes == ys).sum() / len(ys)

        return loss.detach(), accuracy.detach()

    def forward(self, dataloader: DataLoader, optimize: bool = True):
        total_batches = len(dataloader)

        loss_sum = 0
        accuracy_sum = 0

        for batch in dataloader:
            loss, accuracy = self.forward_batch(batch, optimize=optimize)
            loss_sum += loss
            accuracy_sum += accuracy

        return float(loss_sum) / total_batches, float(accuracy_sum) / total_batches

    def train(self,
              epochs: int,
              dataloader: DataLoader,
              plot: bool = False,
              plot_suptitle: Optional[str] = None,
        ):
    
        losses = torch.empty(epochs)
        accuracies = torch.empty(epochs)

        print("Training ", end="")
        for epoch in range(epochs):
            print(".", end="", flush=True)
            loss, accuracy = self.forward(dataloader)
            losses[epoch] = loss
            accuracies[epoch] = accuracy
        print()

        if plot:
            fig = plt.figure()
            if plot_suptitle:
                fig.suptitle(plot_suptitle)
            plt.subplot(1, 2, 1)
            plt.title("Loss")
            plt.plot(range(epochs), losses)
            plt.xlabel("Epoch")

            plt.subplot(1, 2, 2)
            plt.plot(range(epochs), accuracies)
            plt.title("Accuracy")
            plt.xlabel("Epoch")

            fig.show()

        return losses[-1], accuracies[-1]

    def evaluate_model(self, dataloader: DataLoader) -> float:
        loss, accuracy = self.forward(dataloader, optimize=False)
        return accuracy

def partition_dataframe(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    a = df.copy()
    b = a.sample(frac=frac)
    
    a.drop(b.index)
    a = a.reset_index()
    b = b.reset_index()

    return a, b

# The training and test set contains 80% and 20% from the dataset respectively.
training_df, test_df = partition_dataframe(diabetes_df, frac=0.2)

training_dataset_with_y = DiabetesDataset(
    training_df,
    x_include=[col for col in diabetes_df.columns if col not in ("Class")],
    y_column="Class")

test_dataset_with_y = DiabetesDataset(
    test_df,
    x_include=[col for col in diabetes_df.columns if col not in ("Class")],
    y_column="Class")

training_dataset_without_y = DiabetesDataset(
    training_df,
    x_include=[col for col in diabetes_df.columns if col not in ("Class", "Y")],
    y_column="Class")

test_dataset_without_y = DiabetesDataset(
    test_df,
    x_include=[col for col in diabetes_df.columns if col not in ("Class", "Y")],
    y_column="Class")

def train_and_evaluate(training_dataset: Dataset, test_dataset: Dataset, experiment_name: str):
    input_features = len(training_dataset[0][0])

    predictor = ClassPredictor(
        in_features=input_features,
        learning_rate=lr)

    _, training_accuracy = predictor.train(
        epochs,
        DataLoader(training_dataset, batch_size),
        plot=True,
        plot_suptitle=experiment_name)

    test_accuracy = predictor.evaluate_model(
        DataLoader(test_dataset))

    print(f"Experiment: {experiment_name}")
    print(f" - Final training accuracy: {training_accuracy}")
    print(f" - Final test accuracy: {test_accuracy}")

# Visualize the deciles
visualize_deciles()

# Just print, like required in the mmn
print(next(iter(DataLoader(training_dataset_with_y, batch_size))))

# Train and evaluate a predictor for Class that is trained on all fields including Y
train_and_evaluate(training_dataset_with_y, test_dataset_with_y, "Predict Class with Y")

# Train and evaluate a predictor for Class that is trained on all fields except Y
train_and_evaluate(training_dataset_without_y, test_dataset_without_y, "Predict Class without Y")

input()