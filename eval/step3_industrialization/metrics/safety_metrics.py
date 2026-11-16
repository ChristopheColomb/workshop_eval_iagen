"""Step 3 - Métrique de sécurité (Role violation).
"""

from eval.common.deepeval_model import build_deepeval_model
from deepeval.metrics import RoleViolationMetric


def role_violation_metric(
    threshold: float,
    role: str,
) -> RoleViolationMetric:
    """Construit la métrique de respect du rôle."""

    return RoleViolationMetric(
        role=role,
        threshold=threshold,
        model=build_deepeval_model(),
    )
