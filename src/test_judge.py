"""One-question check that DeepEval + the Groq judge work together.
Run:  python src/test_judge.py
"""
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from judge import GroqJudge

judge = GroqJudge()
case = LLMTestCase(
    input="Who bears the cost of intraday borrowing?",
    actual_output="The AMC bears the cost of intraday borrowing.",
    retrieval_context=["2.7. In line with para 11.10 of the Master Circular, cost of "
                       "intraday borrowing, if any, shall be borne by the AMC."],
)
metric = FaithfulnessMetric(model=judge, async_mode=False)
metric.measure(case)
print("score:", metric.score)
print("reason:", metric.reason)