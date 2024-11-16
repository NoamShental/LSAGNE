from typing import Tuple

import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor
import numpy as np


def _inner_product_t(a: Tensor, b:Tensor) -> Tensor:
    return torch.sum(a * b, dim=-1)


def _inner_product_np(a: NDArray[float], b:NDArray[float]) -> float:
    return np.sum(a * b, axis=-1)


def calculate_nearest_points_on_2_lines_t(l1: Tuple[Tensor, Tensor], l2: Tuple[Tensor, Tensor]):
    """
    calculate nearest points on 2 lines, based on the article :
    The Minimum Distance Between Two Lines in n-Space - Michael Bard, Denny Himel
    :param l1: first line, containing 2 points, in the format of (p1+t*d)
    :param l2: second line, it is a tuple of 2 points
    :return: tuple of 2 points p1 and p2, while p1 is on l1, and p2 on l2, and the distance between them is the
             minimum distance between l1 and l2.
    """
    # L1 in the format of : z1 = x0 + xt
    # L2 in the format of : z2 = y0 + ys

    # Unpack the lines
    x0, x = l1
    y0, y = l2

    # Set parameters
    A = _inner_product_t(x, x)
    B = 2 * (_inner_product_t(x0, x) - _inner_product_t(x, y0))
    C = 2 * _inner_product_t(x, y)
    D = 2 * (_inner_product_t(y, y0) - _inner_product_t(y, x0))
    E = _inner_product_t(y, y)

    # Calculate s and t
    s = (2*A*D + B*C) / (C ** 2 - 4 * A * E)
    t = (C * s - B) / (2 * A)

    # Calculate nearest points
    x_nearest = x0 + x * t[:, None]
    y_nearest = y0 + y * s[:, None]
    return x_nearest, y_nearest

def calculate_nearest_points_on_2_lines_np(l1: Tuple[NDArray, NDArray], l2: Tuple[NDArray, NDArray]):
    """
    calculate nearest points on 2 lines, based on the article :
    The Minimum Distance Between Two Lines in n-Space - Michael Bard, Denny Himel
    :param l1: first line, containing 2 points, in the format of (p1+t*d)
    :param l2: second line, it is a tuple of 2 points
    :return: tuple of 2 points p1 and p2, while p1 is on l1, and p2 on l2, and the distance between them is the
             minimum distance between l1 and l2.
    """
    # L1 in the format of : z1 = x0 + xt
    # L2 in the format of : z2 = y0 + ys

    # Unpack the lines
    x0, x = l1
    y0, y = l2

    # Set parameters
    A = _inner_product_np(x, x)
    B = 2 * (_inner_product_np(x0, x) - _inner_product_np(x, y0))
    C = 2 * _inner_product_np(x, y)
    D = 2 * (_inner_product_np(y, y0) - _inner_product_np(y, x0))
    E = _inner_product_np(y, y)

    # Calculate s and t
    s = (2*A*D + B*C) / (C ** 2 - 4 * A * E)
    t = (C * s - B) / (2 * A)

    # Calculate nearest points
    x_nearest = x0 + x * t[:, None]
    y_nearest = y0 + y * s[:, None]
    return x_nearest, y_nearest


# def calculate_nearest_points_on_2_lines_t(l1: Tuple[Tensor, Tensor], l2: Tuple[Tensor, Tensor]):
#     """
#     calculate nearest points on 2 lines, based on the article :
#     The Minimum Distance Between Two Lines in n-Space - Michael Bard, Denny Himel
#     :param l1: first line, containing 2 points, in the format of (p1+t*d)
#     :param l2: second line, it is a tuple of 2 points
#     :return: tuple of 2 points p1 and p2, while p1 is on l1, and p2 on l2, and the distance between them is the
#              minimum distance between l1 and l2.
#     """
#     # L1 in the format of : z1 = x0 + xt
#     # L2 in the format of : z2 = y0 + ys
#
#     # Unpack the lines
#     x0, x = l1
#     y0, y = l2
#
#     # Set parameters
#     A = torch.dot(x, x)
#     B = 2 * (torch.dot(x0, x) - torch.dot(x, y0))
#     C = 2 * torch.dot(x, y)
#     D = 2 * (torch.dot(y, y0) - torch.dot(y, x0))
#     E = torch.dot(y, y)
#
#     # Calculate s and t
#     s = (2*A*D + B*C) / (C ** 2 - 4 * A * E)
#     t = (C * s - B) / (2 * A)
#
#     # Calculate nearest points
#     x_nearest = x0 + x * t
#     y_nearest = y0 + y * s
#     return x_nearest, y_nearest