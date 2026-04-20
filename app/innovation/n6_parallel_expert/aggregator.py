"""N6: Parallel Expert Streaming — Expert output aggregation."""

from __future__ import annotations
import mlx.core as mx
from app.core.logging import get_logger

logger = get_logger(__name__)

class ExpertAggregator:
    """Aggregates expert outputs with routing weights."""

    @staticmethod
    def weighted_sum(
        expert_outputs: list[mx.array],
        routing_weights: mx.array,
    ) -> mx.array:
        """Compute weighted sum of expert outputs.

        Args:
            expert_outputs: List of [batch, hidden_dim] tensors
            routing_weights: [num_experts] weights (already softmax'd by router)

        Returns:
            [batch, hidden_dim] weighted combination
        """
        if len(expert_outputs) == 1:
            return expert_outputs[0] * routing_weights[0]

        result = expert_outputs[0] * routing_weights[0]
        for i in range(1, len(expert_outputs)):
            result = result + expert_outputs[i] * routing_weights[i]
        return result

    @staticmethod
    def top_k_aggregate(
        all_expert_outputs: list[mx.array],
        router_logits: mx.array,
        k: int = 8,
    ) -> mx.array:
        """Select top-k experts and aggregate.

        Args:
            all_expert_outputs: All expert outputs (precomputed or cached)
            router_logits: Raw router output [num_experts]
            k: Number of experts to select

        Returns:
            Weighted combination of top-k experts
        """
        top_k_indices = mx.argpartition(-router_logits, k)[:k]
        top_k_logits = router_logits[top_k_indices]
        weights = mx.softmax(top_k_logits)

        selected = [all_expert_outputs[i.item()] for i in top_k_indices]
        return ExpertAggregator.weighted_sum(selected, weights)
