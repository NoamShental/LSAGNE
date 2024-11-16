from __future__ import annotations
from dataclasses import dataclass

import pandas as pd
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ScatterData2D:
    x_data: NDArray[float]
    x_legend: str
    y_data: NDArray[float]
    y_legend: str
    labels: NDArray[str]
    labels_legend: str = 'label'

    @property
    def unique_labels(self):
        return np.unique(self.labels)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(
                {self.x_legend: self.x_data,
                 self.y_legend: self.y_data,
                 self.labels_legend: self.labels})

    def sort_by_label_importance(self, display_labels_importance: NDArray[str]) -> ScatterData2D:
        importance_np = np.full(len(self.labels), 0)
        for i, importance_label in enumerate(display_labels_importance):
            idx = self.labels == importance_label
            importance_np[idx] = i + 1
        # we wish the important samples to be printed last
        sorted_idx = np.argsort(importance_np)
        return ScatterData2D(
            x_data=self.x_data[sorted_idx],
            x_legend=self.x_legend,
            y_data=self.y_data[sorted_idx],
            y_legend=self.y_legend,
            labels=self.labels[sorted_idx]
        )





class ScatterData3D():
    def __init__(self,
                 x_data,
                 x_legend: str,
                 y_data,
                 y_legend: str,
                 z_data,
                 z_legend: str,
                 labels,
                 labels_legend: str = 'label'):
        self.x_data = x_data
        self.x_legend = x_legend
        self.y_data = y_data
        self.y_legend = y_legend
        self.z_data = z_data
        self.z_legend = z_legend
        self.labels = labels
        self.labels_legend = labels_legend

    @property
    def unique_labels(self):
        return np.unique(self.labels)

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(
                {self.x_legend: self.x_data,
                 self.y_legend: self.y_data,
                 self.z_legend: self.z_data,
                 self.labels_legend: self.labels})
