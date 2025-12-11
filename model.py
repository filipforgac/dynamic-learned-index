import torch
from torch import Tensor
from torch.nn import Module, Linear, ReLU, Sequential
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class MLP(Module):
    def __init__(self, in_features: int, out_features: int):
        super(MLP, self).__init__()
        self.layers = Sequential(
            Linear(in_features, 512),
            ReLU(),
            Linear(512, out_features),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)

    def expand_to(self, n_buckets: int) -> None:
        old_classifier: Module = self.layers[-1]
        n_current_buckets = old_classifier.out_features

        if n_buckets <= n_current_buckets:
            return

        logger.debug(f"[MODEL] Expanding classifier from {n_current_buckets} to {n_buckets} buckets")

        in_features = old_classifier.in_features
        device = old_classifier.weight.device
        dtype = old_classifier.weight.dtype

        new_classifier = Linear(in_features, n_buckets)
        new_classifier.to(device=device, dtype=dtype)

        with torch.no_grad():
            if n_current_buckets > 0:
                new_classifier.weight[:n_current_buckets] = old_classifier.weight[:n_current_buckets]
                new_classifier.bias[:n_current_buckets] = old_classifier.bias[:n_current_buckets]

        self.layers[-1] = new_classifier

    def predict_probabilities(self, inputs: Tensor) -> Tensor:
        logits = self.forward(inputs)
        return F.softmax(logits, dim=-1)
