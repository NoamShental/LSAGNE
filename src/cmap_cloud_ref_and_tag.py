from dataclasses import dataclass
from functools import cached_property

from src.cmap_cloud_ref import CmapCloudRef
from src.models.cmap_cloud_tag import CmapCloudTag


@dataclass(frozen=True, eq=True, unsafe_hash=True)
class CmapCloudRefAndTag:
    cloud_ref: CmapCloudRef
    tag: CmapCloudTag

    @cached_property
    def name(self) -> str:
        return f'{self.cloud_ref.name} ({self.tag.value})'
