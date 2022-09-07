# from ctypes import Union
# Import Union
from typing import Union
import os

from typing import List
import torch
from torch import Tensor
from torch import nn

from .dataset import GraphDataset

__all__ = ["unsorted_segment_sum", "euclidean_feats"]


def unsorted_segment_sum(
    data: Tensor, segment_ids: Tensor, num_segments: int
) -> Tensor:
    r"""Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    Adapted from https://github.com/vgsatorras/egnn.
    """
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result


def euclidean_feats(edge_index: Tensor, x: Tensor, s: Union[Tensor, None]) -> List[Tensor]:
    i, j = edge_index
    x_diff = x[i] - x[j]
    norms = norm(x_diff).unsqueeze(1)
    dots = dot(x[i], x[j]).unsqueeze(1)
    norms, dots = psi(norms), psi(dots)
    
    # Handle first GNN iteration
    if s is not None:
        s_cat = torch.cat([s[i], s[j]], dim=1)
    else:
        s_cat = None

    return norms, dots, x_diff, s_cat


def norm(x: Tensor) -> Tensor:
    r""" Euclidean square norm
         `\|x\|^2 = x[0]^2+x[1]^2+x[2]^2`
    """
    x_sq = torch.pow(x, 2)
    return x_sq.sum(dim=-1)


def dot(x: Tensor, y: Tensor) -> Tensor:
    r""" Euclidean inner product
         `<x,y> = x[0]y[0]+x[1]y[1]+x[2]y[2]`
    """
    xy = x * y
    return xy.sum(dim=-1)


def psi(x: Tensor) -> Tensor:
    """ `\psi(x) = sgn(x) \cdot \log(|x| + 1)`
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1)

def load_dataset(input_dir, data_split):
    graph_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npz')]
    dataset = GraphDataset(graph_files=graph_list)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset[:sum(data_split)], data_split
    )

    return train_dataset, val_dataset, test_dataset

def make_mlp(
    input_size,
    sizes,
    hidden_activation="SiLU",
    output_activation=None,
    layer_norm=False,
    batch_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1], track_running_stats=False, affine=False))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[-1], track_running_stats=False, affine=False))
        layers.append(output_activation())
    return nn.Sequential(*layers)