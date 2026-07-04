"""skill() (F4): package a fitted model / function as a named, reusable, indexed verb."""

import unittest

import numpy as np

from mixle.inference import create
from mixle.inference.skill import Skill, SkillRegistry, default_registry, skill
from mixle.substrate import Substrate


def _scalar_art(seed=0):
    return create([float(x) for x in np.random.RandomState(seed).normal(5, 2, 300)], seed=seed)


class SkillWrappingTest(unittest.TestCase):
    def test_wraps_a_function_into_a_callable_skill(self):
        reg = SkillRegistry()
        sk = skill("greet", lambda name: f"hi {name}", description="greet a user", registry=reg)
        self.assertIsInstance(sk, Skill)
        self.assertEqual(sk("ada"), "hi ada")
        self.assertIn("greet", reg)

    def test_inherits_the_models_certificate(self):
        reg = SkillRegistry()
        sk = skill("spend", _scalar_art(), description="sample spend", registry=reg)
        self.assertIsNotNone(sk.certificate)
        self.assertGreaterEqual(int(sk.guarantee), 4)  # certified model → certified skill

    def test_model_sampler_is_the_default_call(self):
        reg = SkillRegistry()
        sk = skill("spend", _scalar_art(), description="sample spend", registry=reg)
        drawn = list(sk(4, seed=1))
        self.assertEqual(len(drawn), 4)

    def test_explicit_call_overrides(self):
        reg = SkillRegistry()
        sk = skill("half", _scalar_art(), call=lambda x: x / 2, registry=reg)
        self.assertEqual(sk(10), 5)

    def test_cannot_derive_callable_raises(self):
        reg = SkillRegistry()
        with self.assertRaises(TypeError):
            skill("bad", object(), registry=reg)


class RegistryTest(unittest.TestCase):
    def _reg(self):
        reg = SkillRegistry()
        skill(
            "spend_sampler", _scalar_art(), description="sample synthetic customer spend", tags=("spend",), registry=reg
        )
        skill("greet", lambda n: f"hi {n}", description="greet a user by name", tags=("text",), registry=reg)
        return reg

    def test_find_ranks_by_lexical_overlap(self):
        reg = self._reg()
        best = reg.best("sample some spend data")
        self.assertEqual(best.name, "spend_sampler")

    def test_find_returns_empty_on_no_overlap(self):
        reg = self._reg()
        self.assertEqual(reg.find("quantum chromodynamics"), [])
        self.assertIsNone(reg.best("quantum chromodynamics"))

    def test_get_and_len(self):
        reg = self._reg()
        self.assertEqual(len(reg), 2)
        self.assertEqual(reg.get("greet").name, "greet")

    def test_index_mirrors_skills_into_substrate(self):
        reg = self._reg()
        s = Substrate()
        ids = reg.index(s)
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(i.kind == "artifact" for i in s.all()))
        self.assertTrue(any(i.payload.get("skill") == "greet" for i in s.all()))

    def test_default_registry_is_shared(self):
        before = len(default_registry())
        skill("uniq_default_probe", lambda: 1, description="probe")
        self.assertEqual(len(default_registry()), before + 1)


if __name__ == "__main__":
    unittest.main()
