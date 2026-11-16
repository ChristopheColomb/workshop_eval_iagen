"""Step 3 - Métrique déterministe (Json correctness).
"""

from typing import Optional

from pydantic import BaseModel
from eval.common.deepeval_model import build_deepeval_model
from deepeval.metrics import JsonCorrectnessMetric


class ITActionPlan(BaseModel):
    ticket_id: str
    status: str
    next_step: Optional[str] = None


def json_correctness_metric(threshold: float) -> JsonCorrectnessMetric:
    """Construit la métrique de validation du schéma JSON."""

    return JsonCorrectnessMetric(
        expected_schema=ITActionPlan,
        model=build_deepeval_model(),
        threshold=threshold,
        include_reason=True,
    )