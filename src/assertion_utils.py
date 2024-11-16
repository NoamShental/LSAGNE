from typing import Callable, Sized

import torch
from torch import Tensor

from src.configuration import config


# TODO move the debug mode check to decorator

def assert_normalized(x: Tensor, dim: int = 1, epsilon: float = 0.0001) -> None:
    """
    Assert that the given tensor is normalized.
    :param x: The tensor
    :param dim: Dimension to normalize
    :param epsilon: Numeric epsilon
    :return: None, throws an exception if non-normalized.
    """
    if not config.debug_mode:
        return
    assert (torch.abs(torch.norm(x, dim=dim) - 1) > epsilon).sum() == 0, 'This tensor is not normalized'


def assert_same_len(*a: Sized, msg: str = 'Lengths not match') -> None:
    """
    Assert that the given tensors have the same length.
    :param a: List of objects with length.
    :param msg: assert message
    :return: None, throws an exception if length not equal.
    """
    if not config.debug_mode:
        return
    if len(a) == 0:
        return
    first_array_length = len(a[0])
    assert all([len(x) == first_array_length for x in a]), msg


def assert_promise_true(promise: Callable[[], bool], msg: str = 'The promise is not evaluated to true') -> None:
    """
    Assert that the given promise is evaluated to true.
    :param promise: The promise
    :param msg: assert message
    :return: None, throws an exception if condition not met.
    """
    if not config.debug_mode:
        return
    assert promise(), msg

def assert_promise(promise: Callable[[], None]) -> None:
    """
    Assert that the given promise is not raising errors.
    :param promise: The promise
    :return: None, throws an exception if condition not met.
    """
    if not config.debug_mode:
        return
    promise()
