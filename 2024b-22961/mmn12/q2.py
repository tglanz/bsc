from typing import List, Mapping, Optional 
from pprint import pp
import math

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

def add(lhs: MyScalar, rhs: MyScalar) -> MyScalar:
    value = lhs.value + rhs.value
    parents = [lhs, rhs]

    grad = {
            id(lhs): 1,
            id(rhs): 1,
            }

    return MyScalar(value, parents, grad)

def sub(lhs: MyScalar, rhs: MyScalar) -> MyScalar:
    value = lhs.value - rhs.value
    parents = [lhs, rhs]
    grad = {
            id(lhs): 1,
            id(rhs): -1,
            }

    return MyScalar(value, parents, grad)

def mul(lhs: MyScalar, rhs: MyScalar) -> MyScalar:
    value = lhs.value * rhs.value
    parents = [lhs, rhs]
    grad = {
            id(lhs): rhs.value,
            id(rhs): lhs.value
            }
    return MyScalar(value, parents, grad)

def div(dividend: MyScalar, divisor: MyScalar) -> MyScalar:
    assert divisor != 0, "Divisor should be non-zero"
    value = dividend.value * divisor.value
    parents = [dividend, divisor]
    grad = {
            id(dividend): 1 / divisor.value,
            id(divisor): dividend.value
            }
    return MyScalar(value, parents, grad)

def power(base: MyScalar, exponent: float) -> MyScalar:
    value = math.pow(base.value, exponent)
    parents = [base]
    grad = { id(base): exponent * math.pow(value, exponent - 1) }
    return MyScalar(value, parents, grad)

def exp(exponent: MyScalar) -> MyScalar:
    value = math.exp(exponent.value)
    parents = [exponent]
    grad = { id(exponent): value }
    return MyScalar(value, parents, grad)

def cos(theta: MyScalar) -> MyScalar:
    value = math.cos(theta.value)
    parents = [theta]
    grad = { id(theta): -1 * math.sin(theta.value) }
    return MyScalar(value, parents, grad)

def sin(theta: MyScalar) -> MyScalar:
    value = math.sin(theta.value)
    parents = [theta]
    grad = { id(theta): math.cos(theta.value) }
    return MyScalar(value, parents, grad)

def ln(operand: MyScalar) -> MyScalar:
    assert operand.value > 0, "Logarithm operand must be positive"

    value = math.log(operand.value)
    parents = [operand]
    grad = { id(operand): 1 / operand.value }
    return MyScalar(value, parents, grad)


def get_gradient(
        operand: MyScalar,
        aliases: Optional[Mapping[int, str]] = None,
) -> Mapping[str, float]:

    # immediate gradients are the basecase
    grad = dict(operand.grad)

    # run dfs on the graph, compute the derivatives according to the chain rule.
    stack = list(operand.parents)

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

def f():
    a = MyScalar(2)
    b = power(a, 2)
    c = exp(b)

    aliases = {
        id(a): "a",
        id(b): "b",
        id(c): "c"
        }


    grad_c = get_gradient(c, aliases)

    pp(aliases, indent=4)
    pp(grad_c, indent=4)

def g():
    a = MyScalar(1)
    b = MyScalar(2)
    c = add(a, b)


    aliases = {
        id(a): "a",
        id(b): "b",
        id(c): "c"
        }


    grad_c = get_gradient(c, aliases)

    pp(aliases, indent=4)
    pp(grad_c, indent=4)

def test_mul():
    a = MyScalar(1)
    b = MyScalar(2)
    c = mul(a, b)

    aliases = {
        id(a): "a",
        id(b): "b",
        id(c): "c"
        }

    grad_c = get_gradient(c, aliases)

    pp(aliases, indent=4)
    pp(grad_c, indent=4)

def test_addmul():
    a = MyScalar(2)
    b = MyScalar(3)
    c = MyScalar(4)

    d = mul(b, c)
    e = add(a, d)

    aliases = {
        id(a): "a",
        id(b): "b",
        id(c): "c",
        id(d): "d",
        id(e): "e"
        }

    grad = get_gradient(e, aliases)

    pp(aliases, indent=4)
    pp(grad, indent=4)

def test_simple_residual():
    a = MyScalar(2)
    b = MyScalar(3)
    c = MyScalar(4)

    d = mul(a, b)
    e = mul(a, c)

    f = add(d, e)

    aliases = {
        id(a): "a",
        id(b): "b",
        id(c): "c",
        id(d): "d",
        id(e): "e",
        id(f): "f",
        }

    grad = get_gradient(f, aliases)
    print(grad)

def main():
    # test_addmul()
    test_simple_residual()
    # c_a = 218.392
    # c_b = 54.598

if __name__ == '__main__':
    main()
