from __future__ import annotations

from typing import Callable, List, Mapping, Optional, Type 
import math
from random import random
import torch

class MyScalar:

    value: float
    parents: List["MyScalar"]
    grad: Mapping[int, float]

    def __init__(self,
                 value: float,
                 parents: Optional[List["MyScalar"]] = None,
                 grad: Optional[Mapping[int, float]] = None,
    ):
        self.value = value
        self.parents = [] if parents is None else parents
        self.grad = { id(self): 1.0 } if grad is None else grad

    def add(self: MyScalar, other: float) -> MyScalar:
        value = self.value + other 
        parents = [self]

        grad = {
                id(self): 1,
                }

        return MyScalar(value, parents, grad)

    def __add__(self: MyScalar, rhs: MyScalar) -> MyScalar:
        value = self.value + rhs.value
        parents = [self, rhs]

        grad = {
                id(self): 1,
                id(rhs): 1,
                }

        return MyScalar(value, parents, grad)

    def __sub__(self: MyScalar, rhs: MyScalar) -> MyScalar:
        value = self.value - rhs.value
        parents = [self, rhs]
        grad = {
                id(self): 1,
                id(rhs): -1,
                }

        return MyScalar(value, parents, grad)

    def sub(self: MyScalar, rhs: float) -> MyScalar:
        value = self.value - rhs
        parents = [self]

        grad = {
                id(self): 1,
                }

        return MyScalar(value, parents, grad)

    def __mul__(self: MyScalar, rhs: MyScalar) -> MyScalar:
        value = self.value * rhs.value
        parents = [self, rhs]
        grad = {
                id(self): rhs.value,
                id(rhs): self.value
                }
        return MyScalar(value, parents, grad)

    def mul(self: MyScalar, rhs: float) -> MyScalar:
        value = self.value * rhs
        parents = [self]
        grad = {
                id(self): rhs,
                }
        return MyScalar(value, parents, grad)

    def __truediv__(self: MyScalar, divisor: MyScalar) -> MyScalar:
        assert divisor.value != 0, "Divisor should be non-zero"
        value = self.value / divisor.value
        parents = [self, divisor]
        grad = {
                id(self): 1 / divisor.value,
                id(divisor): self.value * -1 / (divisor.value**2)
                }
        return MyScalar(value, parents, grad)

    def div(self: MyScalar, divisor: float) -> MyScalar:
        assert divisor != 0, "Divisor should be non-zero"
        value = self.value * divisor
        parents = [self, divisor]
        grad = {
                id(self): 1 / divisor,
                }
        return MyScalar(value, parents, grad)

    def pow(self: MyScalar, exponent: float) -> MyScalar:
        value = math.pow(self.value, exponent)
        parents = [self]
        grad = { id(self): exponent * math.pow(self.value, exponent - 1) }
        return MyScalar(value, parents, grad)

    def exp(self: MyScalar) -> MyScalar:
        value = math.exp(self.value)
        parents = [self]
        grad = { id(self): value }
        return MyScalar(value, parents, grad)

    def cos(self: MyScalar) -> MyScalar:
        value = math.cos(self.value)
        parents = [self]
        grad = { id(self): -1 * math.sin(self.value) }
        return MyScalar(value, parents, grad)

    def sin(self: MyScalar) -> MyScalar:
        value = math.sin(self.value)
        parents = [self]
        grad = { id(self): math.cos(self.value) }
        return MyScalar(value, parents, grad)

    def log(self: MyScalar) -> MyScalar:
        assert self.value > 0, "Logarithm operand must be positive"

        value = math.log(self.value)
        parents = [self]
        grad = { id(self): 1 / self.value }
        return MyScalar(value, parents, grad)


    def get_gradient(
            self: MyScalar,
            aliases: Optional[Mapping[int, str]] = None,
    ) -> Mapping[str, float]:

        # immediate gradients are the basecase
        grad = dict(self.grad)

        # run dfs on the graph, compute the derivatives according to the chain rule.
        stack = list(self.parents)

        while stack:
            node = stack.pop()

            for parent in node.parents:
                # addititive residuals, multiplicative chain rule 
                grad[id(parent)] = grad.get(id(parent), 0) + (grad[id(node)] * node.grad[id(parent)])
                stack.append(parent)

        ans = {}
        aliases = aliases or {}
        for k, v in (grad).items():
            if k in aliases:
                ans[aliases.get(k, str(k))] = v

        return ans

class Tester:
    @staticmethod
    def compute_torch_grad(initializers, f):
        tensors = [
                torch.tensor(initializer, requires_grad=True)
                for initializer
                in initializers]

        ans = f(*tensors)
        ans.backward()

        grad = {}
        for i, tensor in enumerate(tensors):
            name = chr(ord('a') + i)
            grad[name] = tensor.grad.item()

        return grad

    @staticmethod
    def compute_custom_grad(initializers, f):
        myscalars = [
                MyScalar(initializer)
                for initializer
                in initializers]

        ans = f(*myscalars)

        aliases = {}
        for i, myscalar in enumerate(myscalars):
            name = chr(ord('a') + i)
            aliases[id(myscalar)] = name
        grad = ans.get_gradient(aliases)
        return grad

    @staticmethod
    def error(a, b):
        if max(a, b) == 0:
            return min(a, b)

        return abs(a - b) / max(a, b)

    @staticmethod
    def run(name, argc, f):
        print(f"{name}")

        initializers = [random() for _ in range(argc)]
        custom_grad = Tester.compute_custom_grad(initializers, f)
        torch_grad = Tester.compute_torch_grad(initializers, f)

        assert custom_grad.keys() == torch_grad.keys(), "Torch and Custom grads yielded different leafs"

        for k, custom_val in custom_grad.items():
            torch_val = torch_grad[k]
            error = Tester.error(custom_val, torch_val) 
            print(f" - {k}, error={round(error, 3)}, custom_grad={custom_val}, torch_grad={torch_val}")
            assert error < 0.1, f"Torch and Custom grad[{k}] differ"


def main():
    test_cases = [
            ("exp", 1, lambda a: a.exp()),

            ("a + b", 2, lambda a, b: a + b),
            ("a - b", 2, lambda a, b: a - b),
            ("a * b", 2, lambda a, b: a * b),
            ("a / b", 2, lambda a, b: a / b),

            ("ln", 1, lambda a: a.log()),

            ("large power", 1, lambda a: a.pow(5)),
            ("zero power", 1, lambda a: a.pow(0)),
            ("fractional power", 1, lambda a: a.pow(0.3)),
            ("negative power", 1, lambda a: a.pow(-4.0)),

            ("cos", 1, lambda a: a.cos()),
            ("sin", 1, lambda a: a.sin()),

            ("complex graph, such that a is used in more than one branch",
             4,
             lambda a, b, c, d: ((a.log() / b.sin()) + (a.cos() * c) - (a + d.exp())).pow(2)
             ),

            ("chain operations",
             1,
             lambda a: a.pow(2).cos().sin().sin().exp().pow(3).log() / a + a * a
             )
        ]

    for name, argc, f in test_cases:
        Tester.run(name, argc, f)

if __name__ == '__main__':
    main()
