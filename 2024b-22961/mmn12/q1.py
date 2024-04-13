from typing import List, Sequence
import torch
import matplotlib.pyplot as plt

def sample_from_distribution(dist: List[float]) -> int:
    u = torch.rand([1])[0]
    
    if not (torch.tensor(dist) >= 0).all():
        raise Exception("Invalid distribution: all probabilities must be non-negative")

    if not (torch.tensor(dist).sum() == 1):
        raise Exception("Invalid distribution: measure must be 1")

    acc = 0
    for i, v in enumerate(dist):
        if u <= acc + v:
            return i

        acc += v

    assert False, "Unreachable"

def my_sampler(size: Sequence[int], dist: List[float], requires_grad: bool = False) -> torch.Tensor:
    tensor = torch.empty(size).apply_(lambda _: sample_from_distribution(dist))
    tensor.requires_grad = requires_grad
    return tensor

def create_random_distribution(numel: int) -> List[float]:
    dist = torch.rand((numel, ))
    dist = dist / dist.sum()
    return dist.tolist()

def main():
    k = 10_000
    dist = [0.1, 0.2, 0.7]
    tensor = my_sampler((k, ), dist)

    plt.figure()
    plt.bar(x=[0, 1, 2], height=tensor.histc(3))
    plt.title(f"Histogram of {k} samples from the distribution {dist}")
    plt.xticks([0, 1, 2])

    plt.show()

if __name__ == '__main__':
    main()
