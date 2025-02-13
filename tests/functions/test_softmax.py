import numpy as np
import torch

import .functions as F

from .tensor import Tensor


def test_simple_softmax():
    x = np.array([0, 1, 2], np.float32)

    mx = Tensor(x, requires_grad=True)
    y = F.softmax(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.softmax(tx, dim=0)

    assert np.allclose(y.data, ty.data)

    y.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad, tx.grad, rtol=1e-05, atol=1e-05)


def test_softmax():
    x = np.array([[0, 1, 2], [2, 4, 8]], np.float32)

    mx = Tensor(x, requires_grad=True)
    y = F.softmax(mx)

    tx = torch.tensor(x, requires_grad=True)
    ty = torch.softmax(tx, dim=1)

    assert np.allclose(y.data, ty.data)

    y.sum().backward()
    ty.sum().backward()

    assert np.allclose(mx.grad, tx.grad, rtol=1e-05, atol=1e-05)
