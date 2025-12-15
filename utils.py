import os
from argparse import ArgumentParser

import numpy as np
import logging
import h5py
from pathlib import Path

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--dataset", choices=["laion2B", "agnews-mxbai"], default="laion2B")
    parser.add_argument("--dataset-filename", default="laion2B-en-clip768v2-n=300K.h5", type=str)
    parser.add_argument("--queries-filename", default="public-queries-2024-laion2B-en-clip768v2-n=10k.h5", type=str)
    parser.add_argument("--ground-truth", default="gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5", type=str)
    parser.add_argument("--distance-metric", choices=["inner-product", "l2"], default="inner-product", type=str)
    parser.add_argument("--k-neighbors", default=30, type=int)
    parser.add_argument("--n-probe", default=10, type=int)
    parser.add_argument("--init-after-samples", default=25_000, type=int)
    parser.add_argument("--insert-chunk-size", default=5_000, type=int)
    parser.add_argument("--replay-memory-size", default=0, type=int)
    parser.add_argument("--split-after-inserts", default=25_000, type=int)
    parser.add_argument("--csv-file", default="res.csv", type=str)
    return parser


def get_data_path_for(filename: str) -> str:
    return f"data/{filename}"


def herd_from(X: Tensor, amount: int) -> list[int]:
    """
    An implementation of herding algorithm described in Eq.4 in the article
    Class-Incremental Learning: A Survey (https://arxiv.org/pdf/2302.03648).
    Modified for herding from one class and to return references instead of
    actual data.
    """
    if amount <= 0:
        return []

    class_mean = X.mean(dim=0)
    selected = []
    running_sum = torch.zeros_like(class_mean)

    mask = torch.ones(len(X), dtype=torch.bool, device=X.device)

    for k in range(1, min(amount, len(X)) + 1):
        available = X[mask]
        candidate_means = (available + running_sum) / k
        distances = torch.linalg.vector_norm(class_mean - candidate_means, dim=1)

        closest_local = distances.argmin().item()
        global_idx = torch.where(mask)[0][closest_local].item()

        selected.append(global_idx)
        running_sum += X[global_idx]
        mask[global_idx] = False

    return selected


# Inspired by https://github.com/Coda-Research-Group/LearnedMetricIndex/tree/paper-sisap24-indexing-challenge
def load_LAION_hdf5_embeddings(path: str) -> torch.Tensor:
    logging.info(f"[LOAD] Loading HDF5 embeddings of LAION dataset from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"LAION HDF5 file not found: {path}")

    with h5py.File(path, "r") as f:
        data = f["emb"][:]
        logging.info(f"[LOAD] Loaded LAION 'emb' with shape {data.shape}")
        return torch.tensor(data, dtype=torch.float32)


# Inspired by https://github.com/Coda-Research-Group/LearnedMetricIndex/tree/paper-sisap24-indexing-challenge
def load_LAION_ground_truth(gt_path: str, k: int, should_shift_by_one: bool) -> np.ndarray:
    gt_path = Path(gt_path)
    logger.info(f"[GT] Loading ground truth of LAION dataset from: {gt_path}")

    if not gt_path.exists():
        raise FileNotFoundError(f"LAION ground truth file not found: {gt_path}")

    with h5py.File(gt_path, "r") as f:
        knns = np.array(f["knns"])
    logger.info(f"[GT] Loaded LAION 'knns' with shape {knns.shape}")

    if knns.shape[1] < k:
        raise ValueError(f"LAION GT has only {knns.shape[1]} neighbors per query, cannot eval @k={k}")

    I_true = knns[:, :k]
    if should_shift_by_one:
        I_true = I_true - 1

    return I_true


def load_agnews_mxbai_dataset(path: str, k: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    logging.info(f"[LOAD] Loading HDF5 agnews-mxbai dataset from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"LAION HDF5 file not found: {path}")

    with h5py.File(path, "r") as f:
        data = torch.tensor(f["train"][:])
        queries = torch.tensor(f["test"][:])
        ground_truth = np.array(f["neighbors"])
        logging.info(f"[LOAD] Loaded agnews-mxbai 'train' with shape {data.shape}, 'test' with shape {queries.shape}, 'neighbors' with shape {ground_truth.shape}")
        return data, queries, ground_truth[:, :k]


# Inspired by https://github.com/Coda-Research-Group/LearnedMetricIndex/tree/paper-sisap24-indexing-challenge
def recall_at_k(I_true: np.ndarray, I_pred: np.ndarray, k: int) -> float:
    n_I = I_true.shape[0]

    correct = 0
    for i in range(n_I):
        true_set, pred_set = set(I_true[i]), set(I_pred[i])
        correct += len(true_set & pred_set)

    recall = correct / (n_I * k)
    logger.info(f"[EVAL] Recall@{k} = {recall:.4f}")

    return recall
