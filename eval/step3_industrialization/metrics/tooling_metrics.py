"""Step 3 - Métrique outillage (Tool correctness).
"""

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import ToolCallParams


def tool_correctness_metric(
    threshold: float,
    should_exact_match: bool = True,
    should_consider_ordering: bool = False,
) -> ToolCorrectnessMetric:
    """Construit la métrique de correction des appels d'outils."""

    return ToolCorrectnessMetric(
        threshold=threshold,
        evaluation_params=[
            ToolCallParams.INPUT_PARAMETERS,
        ],
        should_exact_match=should_exact_match,
        should_consider_ordering=should_consider_ordering,
    )
