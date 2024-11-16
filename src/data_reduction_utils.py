from abc import ABC, abstractmethod

import umap
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE

from src.class_encoder import ClassEncoder
from src.cmap_cloud_ref import CmapCloudRef


class DataReductionTool(ABC):

    @property
    @abstractmethod
    def reduction_algo_name(self):
        raise NotImplementedError()

    def to_2d(self, points: NDArray[float], classes: NDArray[CmapCloudRef]) -> NDArray[float]:
        class_encoder = ClassEncoder(classes)
        return self.to_2d_impl(points, class_encoder.np_class_to_encoded_label_vectorize(classes))

    @abstractmethod
    def to_2d_impl(self, points: NDArray[float], classes: NDArray[int]) -> NDArray[float]:
        raise NotImplementedError()


class PCAReductionTool(DataReductionTool):

    @property
    def reduction_algo_name(self):
        return 'pca'

    def to_2d_impl(self, points: NDArray[float], classes: NDArray[int]):
        pca = PCA(n_components=2)
        pca.fit(points)
        return pca.transform(points)


class TsneReductionTool(DataReductionTool):

    @property
    def reduction_algo_name(self):
        return 'tsne'

    def __init__(self, seed, tsne_perplexity=30, tnse_n_iter=2000):
        self.tnse_n_iter = tnse_n_iter
        self.tsne_perplexity = tsne_perplexity
        self.seed = seed

    def to_2d_impl(self, points: NDArray[float], classes: NDArray[int]) -> NDArray[float]:
        return TSNE(
            n_components=2,
            verbose=1,
            perplexity=self.tsne_perplexity,
            n_iter=self.tnse_n_iter,
            random_state=self.seed,
            init='pca',
            learning_rate='auto',
            n_jobs=-1).fit_transform(points)


class UmapReductionTool(DataReductionTool):

    @property
    def reduction_algo_name(self):
        return 'umap'

    def to_2d_impl(self, points: NDArray[float], classes: NDArray[int]):
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(points)
        return embedding

class LDAReductionTool(DataReductionTool):

    @property
    def reduction_algo_name(self):
        return 'LDA'

    def to_2d_impl(self, points: NDArray[float], classes: NDArray[int]):
        reducer = LDA()
        embedding = reducer.fit_transform(points,classes)
        return embedding
