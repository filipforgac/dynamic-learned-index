import math
import logging
from collections import defaultdict

import torch
from torch import Tensor
import numpy as np
import faiss
from torch.utils.data import DataLoader

from data import DLIDataset
from model import MLP
from bucket import Bucket, search_vectors
from utils import herd_from, get_device

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger(__name__)

SEED = 42
MIN_BUCKET_SIZE_FOR_SPLIT = 2 * 40  # Based on faiss kmeans which require roughly cluster_size * 40 samples


class DLI:
    def __init__(self, data_dim, init_after_samples, replay_size):
        self._device = get_device()
        self._data_dim = data_dim
        self._model: MLP | None = None
        self._buckets: dict[int, Bucket] = {}
        self._initialized = False
        self._auto_id = 0

        self._init_after_samples = init_after_samples
        self._replay_size = replay_size

        self._init_vectors: list[Tensor] = []
        self._init_ids: list[int] = []

        self._replay_vectors: list[Tensor] = []
        self._replay_ids: list[int] = []

        self._pending_split_X: list[Tensor] = []
        self._pending_split_y: list[Tensor] = []
        self._pending_split_ids: list[int] = []

    def _run_kmeans(self, k: int, X: Tensor) -> Tensor:
        X_np = X.detach().cpu().numpy().astype(np.float32)
        km = faiss.Kmeans(
            d=self._data_dim,
            k=k,
            seed=SEED,
            verbose=False,
            spherical=True
        )
        km.train(X_np)
        _, labels = km.index.search(X_np, 1)
        return torch.from_numpy(labels[:, 0]).long().to(self._device)

    def _train_model(self, loader: DataLoader, epochs=5, lr=0.001) -> None:
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)

        self._model.train()
        for _ in range(epochs):
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self._device, dtype=torch.float32)
                y_batch = y_batch.to(self._device, dtype=torch.long)
                loss = loss_fn(self._model(X_batch), y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def _refresh_replay(self, exclude_buckets: list[int]) -> None:
        if exclude_buckets is None:
            exclude_buckets = []

        X_replay, y_replay = [], []

        if self._replay_size <= 0:
            logger.info("[REPLAY] No replay memory set")
            self._replay_vectors, self._replay_ids = X_replay, y_replay
            return

        n_buckets = len(self._buckets) - len(exclude_buckets)
        replay_per_bucket = max(1, self._replay_size // n_buckets)

        logger.info(
            f"[REPLAY] Refreshing replay buffer "
            f"(total={self._replay_size}, buckets={n_buckets}, per_bucket={replay_per_bucket})"
        )

        def is_eligible(_id) -> bool:
            return _id in self._buckets and _id not in exclude_buckets

        for bid in filter(is_eligible, self._buckets.keys()):
            bucket = self._buckets[bid]
            X = torch.from_numpy(bucket.vectors)
            idxs = herd_from(X, replay_per_bucket)

            for i in idxs:
                X_replay.append(X[i].clone())
                y_replay.append(bid)

        self._replay_vectors = X_replay
        self._replay_ids = y_replay

        pct = len(self._replay_vectors) / self._replay_size
        logger.info(f"[REPLAY] Final replay amount: {len(self._replay_vectors)} ({pct:.1%})")

    def _initialize(self) -> None:
        X = torch.cat(self._init_vectors).to(self._device)
        n_data = X.shape[0]
        n_buckets = max(2, int(math.sqrt(n_data)))

        logger.info(f"[BOOT] Initializing index with {n_buckets} buckets")
        cluster_ids = self._run_kmeans(n_buckets, X)

        self._model = MLP(self._data_dim, n_buckets).to(self._device)
        loader = DataLoader(DLIDataset(X, cluster_ids), batch_size=256, shuffle=True)
        self._train_model(loader)

        for b in range(n_buckets):
            idxs = (cluster_ids == b).nonzero(as_tuple=True)[0].cpu()
            if len(idxs) == 0:
                continue

            vecs = X[idxs]
            bids = [self._init_ids[i] for i in idxs.tolist()]

            bucket = Bucket(self._data_dim)
            bucket.insert(vecs, bids)
            self._buckets[b] = bucket

        self._initialized = True
        self._init_vectors.clear()
        self._init_ids.clear()

        logger.info("[BOOT] Index initialized successfully")

    def _maybe_split_bucket(self) -> int | None:
        largest_bid = max(self._buckets, key=lambda b: len(self._buckets[b]))
        bucket = self._buckets[largest_bid]
        size = len(bucket)

        if size < MIN_BUCKET_SIZE_FOR_SPLIT:
            logger.debug(f"[SPLIT] Can't split bucket {largest_bid} (size={size}, required={MIN_BUCKET_SIZE_FOR_SPLIT})")
            return None

        logger.debug(f"[SPLIT] Splitting bucket {largest_bid} (size={size})")
        X = torch.from_numpy(bucket.vectors).to(self._device)
        ids = bucket.ids
        cluster_ids = self._run_kmeans(2, X)

        cluster_0 = (cluster_ids == 0).sum().item()
        cluster_1 = (cluster_ids == 1).sum().item()

        larger = 0 if cluster_0 >= cluster_1 else 1
        smaller = 1 - larger

        larger_idx = (cluster_ids == larger).nonzero(as_tuple=True)[0].cpu()
        smaller_idx = (cluster_ids == smaller).nonzero(as_tuple=True)[0].cpu()

        old_bucket = Bucket(self._data_dim)
        old_bucket.insert(X[larger_idx].cpu(), [ids[i] for i in larger_idx.tolist()])
        self._buckets[largest_bid] = old_bucket

        new_bid = max(self._buckets.keys()) + 1
        new_bucket = Bucket(self._data_dim)
        split_ids = [int(ids[i]) for i in smaller_idx.tolist()]
        new_bucket.insert(X[smaller_idx].cpu(), split_ids)
        self._buckets[new_bid] = new_bucket
        logger.debug(
            f"[SPLIT] Created new bucket {new_bid} "
            f"(old bucket size -> {len(old_bucket)}, new bucket size -> {len(new_bucket)})"
        )

        num_classes = len(self._buckets)
        self._model.expand_to(num_classes)
        logger.debug(f"[MODEL] Expanded classifier to {num_classes} buckets")

        self._pending_split_X.append(torch.cat([X[larger_idx], X[smaller_idx]]))
        self._pending_split_y.append(torch.tensor(
            [largest_bid] * len(larger_idx) + [new_bid] * len(smaller_idx),
            dtype=torch.long,
            device=self._device
        ))
        self._pending_split_ids.extend(split_ids)

        return new_bid

    def _maybe_split_buckets(self) -> list[int]:
        num_buckets = len(self._buckets)
        max_splits = max(1, math.ceil(math.log2(num_buckets)))
        logger.info(f"[SPLIT] Splits allowed this batch: {max_splits}")

        new_bucket_ids = []
        for _ in range(max_splits):
            new_bid = self._maybe_split_bucket()
            if new_bid is None:
                break
            new_bucket_ids.append(new_bid)

        return new_bucket_ids

    def _retrain(self, X_inserted: Tensor, predicted_bucket_ids: list[int]) -> None:
        train_X = []
        train_y = []

        train_X.append(X_inserted)
        train_y.append(torch.tensor(predicted_bucket_ids, dtype=torch.long, device=self._device))

        if self._replay_vectors:
            X_rep = torch.stack(self._replay_vectors).to(self._device)
            Y_rep = torch.tensor(self._replay_ids, dtype=torch.long, device=self._device)
            train_X.append(X_rep)
            train_y.append(Y_rep)

        train_X.extend(self._pending_split_X)
        train_y.extend(self._pending_split_y)

        X_train = torch.cat(train_X)
        y_train = torch.cat(train_y)

        logger.info(f"[TRAIN] Retraining on {len(X_train)} samples")
        loader = DataLoader(DLIDataset(X_train, y_train), batch_size=256, shuffle=True)
        self._train_model(loader)

    def insert(self, vectors: Tensor) -> None:
        """NOTE: insert(...) IS NOT thread-safe and must not be executed concurrently."""
        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(0)

        vectors = vectors.to(self._device)
        n_data = vectors.shape[0]
        new_ids = list(range(self._auto_id, self._auto_id + n_data))
        data_dim = vectors.shape[1]
        assert data_dim == self._data_dim, "Inserted data dimension doesn't match indexed dimension"

        if not self._initialized:
            self._init_vectors.append(vectors)
            self._init_ids.extend(new_ids)
            self._auto_id += n_data

            if sum(v.shape[0] for v in self._init_vectors) >= self._init_after_samples:
                self._initialize()

            return

        with torch.no_grad():
            probs = self._model.predict_probabilities(vectors)
        _, predicted_bucket_ids = torch.max(probs, dim=1)

        per_bucket_vectors = defaultdict(list)
        per_bucket_ids = defaultdict(list)

        for i in range(n_data):
            vid = new_ids[i]
            bid = int(predicted_bucket_ids[i])
            per_bucket_vectors[bid].append(vectors[i].cpu())
            per_bucket_ids[bid].append(vid)

        for bid, vecs in per_bucket_vectors.items():
            self._buckets[bid].insert(torch.stack(vecs), per_bucket_ids[bid])
        self._auto_id += n_data

        new_buckets = self._maybe_split_buckets()
        was_not_split_mask = [vid not in self._pending_split_ids for vid in new_ids]
        X_inserted_not_split = vectors[was_not_split_mask]
        not_split_predicted_bucket_ids = [int(b) for was_not_split, b in zip(was_not_split_mask, predicted_bucket_ids) if was_not_split]

        self._refresh_replay(exclude_buckets=new_buckets)
        self._retrain(X_inserted_not_split, not_split_predicted_bucket_ids)  # vectors in the newly created buckets will be trained from pending splits
        self._pending_split_X, self._pending_split_y, self._pending_split_ids = [], [], []

    def _visit_buckets(
        self,
        k: int,
        predicted_buckets,
        query: Tensor,
        qid: int,
        distance_metric: str,
    ) -> tuple[Tensor, Tensor, int]:
        n_buckets = len(predicted_buckets)
        D = torch.empty((n_buckets * k,), dtype=torch.float32)
        I = torch.empty((n_buckets * k,), dtype=torch.int32)

        for i, bid in enumerate(predicted_buckets):
            I_bucket, D_bucket = self._buckets[bid].search(query, k, distance_metric)
            start = i * k
            D[start:start + k] = D_bucket.view(-1)
            I[start:start + k] = I_bucket.view(-1)

        D_top, idx_top = torch.topk(D, k, largest=False)
        return D_top, I[idx_top], qid

    def _search_init_vectors(self, queries: np.ndarray, k: int, distance_metric: str) -> tuple[np.ndarray, np.ndarray]:
        assert self._init_vectors, "No vectors to search"

        n_queries = len(queries)
        D = np.empty((n_queries, k), dtype=np.float32)
        I = np.empty((n_queries, k), dtype=np.int32)

        vectors_np = np.array(self._init_vectors)
        ids_np = np.array(self._init_ids)

        with ThreadPoolExecutor(max_workers=9) as executor:
            results = executor.map(
                lambda q: (
                    search_vectors(
                        vectors_np,
                        ids_np,
                        torch.from_numpy(queries[q:q + 1]),
                        k,
                        distance_metric,
                    ),
                    q,
                ),
                range(n_queries),
            )

            for dists, ids, qid in tqdm(results, total=n_queries):
                D[qid] = dists.numpy()
                I[qid] = ids.numpy()

        return D, I

    def search(self, queries: Tensor, k: int, n_probe: int, distance_metric: str) -> tuple[np.ndarray, np.ndarray]:
        """NOTE: search(...) IS thread-safe and can be executed concurrently."""
        if queries.dim() == 1:
            queries = queries.unsqueeze(0)

        data_dim = queries.shape[1]
        assert data_dim == self._data_dim, "Queries dimension doesn't match indexed dimension"

        queries_np = queries.cpu().numpy().astype(np.float32)
        if not self._initialized:
            return self._search_init_vectors(queries_np, k, distance_metric)

        with torch.no_grad():
            probs = self._model.predict_probabilities(queries.to(self._device))

        n_probe = min(n_probe, len(self._buckets))
        _, predicted_bucket_ids = torch.topk(probs, n_probe, dim=1)

        n_queries = len(queries_np)
        D = np.empty((n_queries, k), dtype=np.float32)
        I = np.empty((n_queries, k), dtype=np.int32)

        torch.set_num_threads(3)
        faiss.omp_set_num_threads(3)

        with ThreadPoolExecutor(max_workers=9) as executor:
            results = executor.map(
                lambda q: self._visit_buckets(
                    k,
                    predicted_bucket_ids[q].cpu().tolist(),
                    torch.from_numpy(queries_np[q:q + 1]),
                    q,
                    distance_metric,
                ),
                range(n_queries)
            )

            for dists, ids, qid in tqdm(results, total=n_queries):
                D[qid] = dists.numpy()
                I[qid] = ids.numpy()

        return D, I


def insert_data_in_chunks(index: DLI, data: Tensor, chunk_size: int) -> None:
    data_size = data.shape[0]
    logger.info(f"[INSERT] Inserting in chunks of {chunk_size} (total {data_size})")

    for start in range(0, data_size, chunk_size):
        end = min(start + chunk_size, data_size)
        chunk = data[start:end]

        logger.debug(f"[INSERT] Inserting rows {start}–{end}")
        index.insert(chunk)

        logger.info(f"[INSERT] Progress: {end}/{data_size} inserted")
