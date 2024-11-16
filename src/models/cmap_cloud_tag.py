from __future__ import annotations

from enum import Enum


class CmapCloudTag(Enum):
    REAL_TRAINED = 'Real Trained'
    PREDICTED_TRAINED = 'Predicted Trained'
    TRAINING_CONCEALED = 'Training Concealed'
    LEFT_OUT = 'Left Out'
    PREDICTED_LEFT_OUT = 'Predicted Left Out'
    CV = 'CV'
    PREDICTED_CV = 'Predicted CV'
    AUGMENTED = 'AUGMENTED'
