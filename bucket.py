import faiss
import numpy as np
import torch
from torch import Tensor


class Bucket:
    def __init__(self, dim: int):
        self._dim = dim
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.ids = np.empty((0,), dtype=np.int32)

    def __len__(self):
        return self.vectors.shape[0]

    def insert(self, data: Tensor, ids):
        if isinstance(ids, int):
            ids = [ids]

        data_np = data.detach().cpu().numpy().astype(np.float32)
        ids_np = np.array(ids, dtype=np.int32)

        self.vectors = np.vstack([self.vectors, data_np])
        self.ids = np.concatenate([self.ids, ids_np])

    def search(self, query: Tensor, k: int, distance_metric: str):
        if len(self.vectors) == 0:
            I = torch.full((1, k), -1, dtype=torch.long)
            D = torch.full((1, k), float("inf"), dtype=torch.float32)
            return I, D

        query_np = query.detach().cpu().numpy().astype(np.float32)

        match distance_metric:
            case "inner-product":
                metric = faiss.METRIC_INNER_PRODUCT
            case "l2":
                metric = faiss.METRIC_L2
            case _:
                raise ValueError("Invalid distance metric")

        D, I_local = faiss.knn(query_np, self.vectors, k, metric=metric)
        I_global = self.ids[I_local]
        return torch.from_numpy(I_global), torch.from_numpy(D)
