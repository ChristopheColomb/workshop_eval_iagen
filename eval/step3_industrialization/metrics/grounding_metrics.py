"""Step 3 - Métrique de grounding (Faithfulness).
"""

from eval.common.deepeval_model import build_deepeval_model
from deepeval.metrics import FaithfulnessMetric


def faithfulness_metric(
    threshold: float,
    truths_extraction_limit: int = 4,
) -> FaithfulnessMetric:
    """Construit la métrique de fidélité au contexte récupéré."""

    return FaithfulnessMetric(
        threshold=threshold,
        model=build_deepeval_model(),
        truths_extraction_limit=truths_extraction_limit,
        include_reason=True,
    )
