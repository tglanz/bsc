import torch
from typing import Sequence, Tuple

def expand(tensor: torch.Tensor, size: torch.Size) -> torch.Tensor:
    """
    Expand a given `tensor`, to a newly allocated tensor, with the shape given by `size`.
    
    The expansion is done by the broadcast rules:
    - As long as the rank of `tensor` is less than `len(size)`
      - Add degenerative axis to the tensor
    - Iterating over every dimesnion `dim`, right to left:
      - If the `tensor.shape[dim] != size[dim]`:
        - If `tensor.shape[dim] == 1`:
          - Concatenate the tensor with itself, `shape[dim]` times, along the `dim`th axis
        - Else: An error is raised indicating that the tensor is not expandable to size `size`

    Arguments
      tensor {torch.Tensor} The tensor to expand
      size {torch.Size} The size to expand the tensor to

    Returns
      {torch.Tensor} A newly allcoated tensor
    """
    ans = tensor.clone()
    rank = len(size)
    
    # Iterate on every dimension
    for i in range(rank):

        # A cursor to the current axis in `size`
        j = rank - 1 - i

        # A cursor to the current dimension in `tensor.shape`.
        # Note that it can be negative at first, but at most -1.
        k = len(ans.shape) - 1 - i

        if k < 0:
            # It is sufficient to only add one degenerate dimension because we
            # add them one by one - so we know that the index will be in bounds.
            ans = ans.unsqueeze(0)
            k = 0

        ans_dimsize = ans.shape[k]
        dimsize = size[j]

        if ans_dimsize != dimsize:
            if ans_dimsize != 1:
                raise Exception(f"Cannot expand tensor with shape {tuple(ans.shape)} to shape {size}")
            ans = torch.cat([ans] * dimsize, dim=k)
        
    return ans

def test_expand():
    """Tester for `are_broadcastable_together`"""
    
    testcases = [
        (torch.ones((1, )), (3,), torch.ones((3,))),
        (torch.empty((1, 2)), (2, 1), None),
        (torch.arange(2), (2, 2), torch.tensor([
            [0, 1],
            [0, 1],
        ])),
        (torch.arange(4).reshape((2, 1, 2)), (2, 2, 2), torch.tensor([
            [[[0, 1]],
             [[1, 2]]],
        ])),
    ]
    
    for i, (tensor, size, expected_tensor) in enumerate(testcases):
        if i == 3:
            print("now")

        print(f"====== Test {i} =======")
        print("tensor", tensor)
        print("size", size)
        print("expected_tensor", expected_tensor)

        try:
            actual_tensor = expand(tensor, size)
            print("actual_tensor", actual_tensor)
        except:
            if expected_tensor is not None:
                assert False, "did not expect expand to raise an error"
            continue

        if expected_tensor is None:
            assert False, "expected expand to raise an error"
        assert torch.equal(actual_tensor, expected_tensor), "expanded tensor is not as expected"

        print("")
                

def are_broadcastable_together(a: torch.Tensor, b: torch.Tensor) -> Tuple[bool, torch.Size]:
    shape_a = list(a.shape)
    shape_b = list(b.shape)
    
    rank_a = len(shape_a)
    rank_b = len(shape_b)

    for i in range(max(rank_a, rank_b)):
        axis_a = rank_a - 1 - i
        axis_b = rank_b - 1 - i

        if axis_a < 0:
            shape_a.insert(0, 1)
            axis_a = 0

        if axis_b < 0:
            shape_b.insert(0, 1)
            axis_b = 0

        size_a = shape_a[axis_a]
        size_b = shape_b[axis_b]

        if size_a != size_b:
            if size_a == 1:
                shape_a[axis_a] = size_b
            elif size_b == 1:
                shape_b[axis_b] = size_a
            else:
                return (False, None)

    assert shape_a == shape_b
    return (True, tuple(shape_a))

def test_are_broadcastable_together():
    
    testcases = [
        ((1,), (1, 1), True, (1, 1)),

        ((2, 1, 2), (2, 1, 1, 3), False, None),
        ((2, 1, 2), (5, 1), True, (2, 5, 2)),
        ((2, 1, 2), (1, 5), False, None),
        ((2, 1, 2), (1,), True, (2, 1, 2)),
        ((2, 1, 2), (5, 3), False, None),

        ((2, 1, 1, 3), (5, 1), True, (2, 1, 5, 3)),
        ((2, 1, 1, 3), (1, 5), False, None),
        ((2, 1, 1, 3), (1,), True, (2, 1, 1, 3)),
        ((2, 1, 1, 3), (5, 3), True, (2, 1, 5, 3)),

        ((5, 1), (1, 5), True, (5, 5)),
        ((5, 1), (1, ), True, (5, 1)),
        ((5, 1), (5, 3), True, (5, 3)),

        ((1, 5), (1, ), True, (1, 5)),
        ((1, 5), (5, 3), False, None),
        ((1, ), (5, 3), True, (5, 3)),
    ]
    
    for a_shape, b_shape, expected_broadcastable, expected_shape in testcases:
        print("a_shape", a_shape)
        print("b_shape", b_shape)

        a = torch.empty(a_shape)
        b = torch.empty(b_shape)
        actual_broadcastable, actual_shape = are_broadcastable_together(a, b)

        print(f"expected_broadcastable={expected_broadcastable}, actual_broadcastable={actual_broadcastable}")
        print(f"expected_shape={expected_shape}, actual_shape={actual_shape}")
        assert expected_broadcastable == actual_broadcastable
        assert expected_shape == actual_shape

        print("")


def main():
    test_expand()
    test_are_broadcastable_together()

if __name__ == '__main__':
    main()