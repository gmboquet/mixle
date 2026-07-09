"""orchestrate(): plan/execute/re-plan loop against a world, stopping on STOP/budget/confidence/done
(workstream C3)."""

import unittest

from mixle.task.orchestrate import orchestrate


class _GridWorld:
    """A tiny deterministic world: an agent on a 1-D line trying to reach ``target``.

    ``step({"tool": "move", "args": {"dx": n}})`` moves; ``move`` with ``dx == 0`` raises (an invalid,
    atypical action) to exercise the re-plan path. ``score()`` is the final distance to target
    (0 = solved). ``done`` is a SEPARATE, explicitly-set flag (not implied by reaching the target) so
    tests can exercise "the plan model says STOP" and "the world reports itself done" independently --
    they are two different termination signals in the real orchestrator contract."""

    def __init__(self, target: int, *, done: bool = False) -> None:
        self.target = target
        self.position = 0
        self._done = done

    def step(self, action):
        if action["tool"] != "move":
            raise ValueError(f"unknown tool {action['tool']!r}")
        dx = action["args"]["dx"]
        if dx == 0:
            raise ValueError("dx=0 is not a valid move")
        self.position += dx
        return {"position": self.position}

    @property
    def done(self) -> bool:
        return self._done

    def score(self):
        return abs(self.target - self.position)


def _greedy_planner(target: int, *, poison_first_step: bool = False):
    """A scripted plan_model: always steps 1 toward the target; ``poison_first_step`` makes the very
    first proposed step an invalid ``dx=0`` to exercise re-planning."""
    calls = {"n": 0}

    def plan(question, history):
        calls["n"] += 1
        position = 0
        for entry in reversed(history):
            if "result" in entry:
                position = entry["result"]["position"]
                break
        if position == target:
            return None  # STOP: already there
        if poison_first_step and calls["n"] == 1:
            return {"tool": "move", "args": {"dx": 0}}
        step = 1 if target > position else -1
        return {"tool": "move", "args": {"dx": step}}

    return plan


class OrchestrateTest(unittest.TestCase):
    def test_reaches_the_target_and_stops_on_plan_stop(self):
        world = _GridWorld(target=3)
        result = orchestrate("reach 3", _greedy_planner(3), world, budget=10)
        self.assertEqual(result.answer, 0)  # distance to target == 0: solved
        self.assertEqual(result.stopped_reason, "plan_stop")
        self.assertEqual(len(result.trace.steps), 3)
        self.assertTrue(all(s.tool == "move" for s in result.trace.steps))

    def test_stops_on_budget_exhaustion_before_reaching_the_target(self):
        world = _GridWorld(target=100)
        result = orchestrate("reach 100", _greedy_planner(100), world, budget=5)
        self.assertEqual(result.stopped_reason, "budget_exhausted")
        self.assertEqual(len(result.trace.steps), 5)
        self.assertEqual(result.answer, 95)  # honest partial progress, not a fabricated success

    def test_reruns_the_plan_model_to_recover_from_a_failed_step(self):
        world = _GridWorld(target=2)
        result = orchestrate("reach 2", _greedy_planner(2, poison_first_step=True), world, budget=10)
        # the poisoned dx=0 step is recorded as a failure, then the retry succeeds and progress continues
        self.assertEqual(result.trace.steps[0].result, {"error": "dx=0 is not a valid move"})
        self.assertEqual(result.answer, 0)
        self.assertEqual(result.stopped_reason, "plan_stop")

    def test_stops_immediately_on_explicit_stop(self):
        world = _GridWorld(target=5)  # not yet reached, not done -- the plan model alone decides to stop
        result = orchestrate("noop", lambda q, h: None, world, budget=10)
        self.assertEqual(result.stopped_reason, "plan_stop")
        self.assertEqual(len(result.trace.steps), 0)

    def test_stops_on_low_confidence(self):
        world = _GridWorld(target=10)

        def unsure_plan(question, history):
            return {"tool": "move", "args": {"dx": 1}, "confidence": 0.1}

        result = orchestrate("reach 10", unsure_plan, world, budget=10, confidence_threshold=0.5)
        self.assertEqual(result.stopped_reason, "low_confidence")
        self.assertEqual(len(result.trace.steps), 0)

    def test_stops_when_the_world_is_already_done(self):
        world = _GridWorld(target=5, done=True)  # far from target, but the world itself is already over
        result = orchestrate("noop", _greedy_planner(5), world, budget=10)
        self.assertEqual(result.stopped_reason, "world_done")
        self.assertEqual(len(result.trace.steps), 0)


if __name__ == "__main__":
    unittest.main()
