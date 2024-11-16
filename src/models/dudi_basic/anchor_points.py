from dataclasses import dataclass
from typing import Dict, TypeVar, Generic

from src.cmap_cloud_ref import CmapCloudRef
from src.models.dudi_basic.perturbations_anchors import PerturbationsAnchors
from src.perturbation import Perturbation

T = TypeVar('T')


@dataclass(frozen=True)
class AnchorPoints(Generic[T]):
    # TODO fix to be pert -> cloud_refs
    perturbation_to_anchors: Dict[Perturbation, PerturbationsAnchors[T]]
    cloud_ref_to_cloud_center: Dict[CmapCloudRef, T]
