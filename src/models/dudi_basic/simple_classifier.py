from typing import List, Type, Optional, TypeVar, Generic, Collection, Dict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.types import Device

from src.class_encoder import ClassEncoder
from src.cmap_cloud_ref import CmapCloudRef
from src.module_init import xavier_uniform_init

T = TypeVar('T')


class SimpleClassifier(nn.Module, Generic[T]):
    def __init__(
            self,
            input_dim: int,
            inner_dims: List[int],
            all_classes: Collection[T],
            class_to_class_weights: Optional[Dict[T, float]],
            layer_type: Type[nn.Linear],
            reduction='mean'
    ):
        super().__init__()
        self.all_classes = all_classes
        n_classes = len(all_classes)
        self.loss_reduction = reduction
        self._class_encoder = ClassEncoder(all_classes)
        self._cross_entropy_loss = self._create_cross_entropy_loss(class_to_class_weights, None, reduction, self._class_encoder.class_to_encoded_label)

        if len(inner_dims) == 0:
            classifier = [layer_type(input_dim, n_classes)]
        else:
            classifier = [layer_type(input_dim, inner_dims[0])]
            for i in range(1, len(inner_dims)):
                classifier.append(layer_type(inner_dims[i - 1], inner_dims[i]))
                # if i != len(inner_dims) - 1:
                # classifier.append(nn.PReLU())
            classifier.append(layer_type(inner_dims[-1], n_classes))
            # Dudi's code:
            # self.y_output = Dense(config.config_map['classes_count'], name='classifier_output', activation='softmax',
            #                       kernel_initializer=config.config_map['initialization_method'],
            #                       kernel_constraint=maxnorm(self.max_weight))(current_layer)
        # classifier.append(nn.PReLU())
        # classifier.append(nn.Softmax(dim=1))

        self.classifier = nn.Sequential(*classifier)
        self.init_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _create_cross_entropy_loss(
            class_to_class_weights: Optional[Dict[T, float]],
            device: Device,
            loss_reduction: str,
            class_to_encoded_label: Dict[CmapCloudRef, int]
    ) -> CrossEntropyLoss:
        if class_to_class_weights is None:
            class_weights = None
        else:
            class_weights_array = np.zeros(len(class_to_class_weights))
            for i, (cls, class_weight) in enumerate(class_to_class_weights.items()):
                class_weights_array[class_to_encoded_label[cls]] = class_weight
            class_weights = torch.tensor(class_weights_array, device=device)
        return CrossEntropyLoss(weight=class_weights, reduction=loss_reduction)

    def forward(self, x):
        return self.classifier(x)

    def loss_fn(self, y_pred, y_true: NDArray[T]):
        # loss = CrossEntropyLoss()
        # return F.cross_entropy(y_pred, y_true)
        y_true = torch.tensor(self._class_encoder.np_class_to_encoded_label_vectorize(y_true), device=self.device, dtype=torch.int64)
        return self._cross_entropy_loss(y_pred, y_true)

    def update_class_weights_array(self, class_to_class_weights: Optional[Dict[T, float]]):
        self._cross_entropy_loss = self._create_cross_entropy_loss(
            class_to_class_weights,
            self.device,
            self.loss_reduction,
            self._class_encoder.class_to_encoded_label
        )

    def init_weights(self):
        xavier_uniform_init(self.classifier)

    def convert_encoded_label_to_class(self, encoded_labels: NDArray[int]) -> NDArray[T]:
        return self._class_encoder.np_encoded_label_to_class_vectorize(encoded_labels)

    def convert_class_to_encoded_label(self, classes: NDArray[T]) -> NDArray[T]:
        return self._class_encoder.np_class_to_encoded_label_vectorize(classes)

    # def __str__(self):
    #     return f'Classifier is {self.classifier}.'
