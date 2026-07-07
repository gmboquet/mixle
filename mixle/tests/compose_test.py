"""compose: chain two models/teachers into one ledger-carrying callable (CARD COMPOSE-a).

Neither stage answers the composite x -> z question alone: stage_a classifies a raw reading into a
category (it knows nothing about actions); stage_b maps a category to a recommended action (it cannot
accept a raw numeric reading). Only the composition bridges x -> z.
"""

import unittest

from mixle.task.compose import ComposedModel, compose


def classify_reading(x: float) -> str:
    """x -> y: a raw sensor reading classified into a severity category. Knows nothing about actions."""
    if x < 30:
        return "low"
    if x < 70:
        return "mid"
    return "high"


_ACTION_TABLE = {"low": "log", "mid": "notify", "high": "escalate"}


def recommend_action(y: str) -> str:
    """y -> z: a category mapped to a recommended action. Its domain is categories, not raw readings."""
    return _ACTION_TABLE[y]


class ComposeTest(unittest.TestCase):
    def test_neither_stage_answers_x_to_z_alone(self):
        # stage_a's own output IS the category, not an action -- it cannot answer the x->z question
        self.assertEqual(classify_reading(87.0), "high")
        # stage_b's domain is categories: feeding it the raw reading directly is a type/domain error
        with self.assertRaises(KeyError):
            recommend_action(87.0)  # not a valid category key

    def test_composition_answers_x_to_z(self):
        pipeline = compose(classify_reading, recommend_action)
        self.assertIsInstance(pipeline, ComposedModel)
        self.assertEqual(pipeline(87.0), "escalate")
        self.assertEqual(pipeline(15.0), "log")
        self.assertEqual(pipeline(50.0), "notify")

    def test_receipt_attributes_the_answer_to_both_stages(self):
        pipeline = compose(classify_reading, recommend_action, name_a="classify", name_b="recommend")
        answer, receipt = pipeline.answer(87.0)
        self.assertEqual(answer, "escalate")
        self.assertEqual(receipt["answer"], "escalate")
        self.assertEqual(len(receipt["stages"]), 2)
        stage1, stage2 = receipt["stages"]
        self.assertEqual(stage1, {"name": "classify", "input": 87.0, "output": "high"})
        self.assertEqual(stage2, {"name": "recommend", "input": "high", "output": "escalate"})
        # the intermediate value chains: stage1's output IS stage2's input
        self.assertEqual(stage1["output"], stage2["input"])

    def test_default_stage_names(self):
        pipeline = compose(classify_reading, recommend_action)
        _answer, receipt = pipeline.answer(10.0)
        self.assertEqual(receipt["stages"][0]["name"], "stage_a")
        self.assertEqual(receipt["stages"][1]["name"], "stage_b")

    def test_three_severities_all_bridge_correctly(self):
        pipeline = compose(classify_reading, recommend_action)
        for x, expected in [(5.0, "log"), (45.0, "notify"), (99.0, "escalate")]:
            self.assertEqual(pipeline(x), expected)


if __name__ == "__main__":
    unittest.main()
