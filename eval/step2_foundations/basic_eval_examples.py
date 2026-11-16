"""Step 2 : deux familles d'évaluation avec DeepEval.

- LLM-as-a-Judge (GEval) : un LLM note la réponse selon un critère libre.
- Grounding (FaithfulnessMetric) : vérifie que la réponse reste fidèle au
  contexte de retrieval (détection d'hallucination).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.common.deepeval_model import build_deepeval_model
from app.rag_agent import SupportRAGAgent

from deepeval import evaluate
from deepeval.evaluate.configs import DisplayConfig
from deepeval.metrics import FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase
from deepeval.test_case import SingleTurnParams


def get_agent_answer_and_context(question: str) -> tuple[str, list[str]]:
    agent = SupportRAGAgent()
    result = agent.answer(question)
    # On renvoie aussi le contexte de retrieval : indispensable pour la faithfulness.
    return result.answer, [item.document.content for item in result.retrieved_documents]


def log_io(question: str, answer: str, retrieval_context: list[str] | None = None) -> None:
    """Affiche l'input et l'output de l'agent pour faciliter le debug."""
    print("\n--- INPUT / OUTPUT ---")
    print(f"[INPUT]  {question}")
    print(f"[OUTPUT] {answer}")
    if retrieval_context is not None:
        print("[CONTEXT]")
        for idx, chunk in enumerate(retrieval_context, start=1):
            print(f"  ({idx}) {chunk}")
    print("--- END INPUT / OUTPUT ---")


def llm_judge_section(verbose: bool = False) -> list[bool]:
    print("\n=== SECTION LLM-AS-A-JUDGE ===")
    model = build_deepeval_model()

    question = "Le VPN est indisponible, que dois-je faire ?"

    answer, _ = get_agent_answer_and_context(question)

    if verbose:
        log_io(question, answer)

    tone_score = GEval(
        name="ProfessionalTone",
        model=model,
        criteria=(
            "Évalue si la réponse adopte un ton professionnel, courtois et respectueux, "
            "adapté à un contexte de support informatique. "
            "La réponse doit être claire, bienveillante et rassurante, "
            "sans familiarité excessive, agressivité, jugement ou formulation méprisante. "
            "Elle doit s'adresser à l'utilisateur de manière polie et utiliser "
            "un langage accessible, sans jargon technique inutile. "
            "Une réponse qui respecte ces critères obtient un bon score."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.5,
    )

    formula_score = GEval(
        name="AnswerStructure",
        model=model,
        criteria=(
            "Évalue la structure de la réponse. "
            "La réponse doit commencer par une formule d'introduction adaptée "
            "qui accueille la demande ou reformule brièvement le problème de l'utilisateur. "
            "Elle doit ensuite apporter une réponse structurée et compréhensible. "
            "Enfin, elle doit se terminer par une formule de conclusion appropriée, "
            "proposant une aide complémentaire ou invitant l'utilisateur à revenir "
            "si le problème persiste. "
            "Vérifie explicitement la présence de ces deux éléments : "
            "une introduction au début et une conclusion à la fin. "
            "Une réponse sans introduction ou sans conclusion doit être pénalisée."
        ),
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.5,
    )

    case_tone = LLMTestCase(input=question, actual_output=answer)
    case_formula = LLMTestCase(input=question, actual_output=answer)

    tone_result = evaluate(test_cases=[case_tone], metrics=[tone_score], display_config=DisplayConfig(print_results=True))
    formula_result = evaluate(test_cases=[case_formula], metrics=[formula_score], display_config=DisplayConfig(print_results=True))

    tone_ok = bool(tone_result.test_results[0].success)
    formula_ok = bool(formula_result.test_results[0].success)
    return [tone_ok, formula_ok]


def grounding_faithfulness_section(verbose: bool = False) -> list[bool]:
    print("\n=== SECTION GROUNDING / FAITHFULNESS ===")
    model = build_deepeval_model()

    question = "Que faire en cas de problème de VPN ?"
    # Variante à tester : "Que faire si la connexion VPN échoue ?"
    
    answer, retrieval_context = get_agent_answer_and_context(question)

    if not retrieval_context:
        raise ValueError("Faithfulness nécessite un contexte documentaire récupéré par l'agent.")
    if verbose:
        log_io(question, answer, retrieval_context)

    # threshold = seuil de réussite : le cas passe si le score >= threshold.
    faithfulness_score = FaithfulnessMetric(threshold=0.5, model=model, penalize_ambiguous_claims=True)

    case = LLMTestCase(input=question, actual_output=answer, retrieval_context=retrieval_context)
    result = evaluate(test_cases=[case], metrics=[faithfulness_score], display_config=DisplayConfig(print_results=True))
    return [bool(result.test_results[0].success)]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2 foundations: judge, grounding.")
    parser.add_argument(
        "--section",
        choices=["all", "judge", "grounding"],
        default="all",
        help="Execute une seule section ou tout le script.",
    )
    parser.add_argument(
        "--log-io",
        action="store_true",
        help="Affiche l'input et l'output de l'agent pour chaque cas de test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    results: list[bool] = []
    if args.section in {"all", "judge"}:
        results.extend(llm_judge_section(verbose=args.log_io))
    if args.section in {"all", "grounding"}:
        results.extend(grounding_faithfulness_section(verbose=args.log_io))

    passed = sum(1 for item in results if item)
    total = len(results)
    print(f"\nSummary: {passed}/{total} checks OK")
    print("Note: des FAIL sont attendus en workshop pour montrer la valeur de l'evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
