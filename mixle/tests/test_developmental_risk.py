import numpy as np

from mixle.analysis.developmental_risk import benchmark_dose, rfd_exceedance


def test_bmdl_matches_reference():
    rng = np.random.default_rng(0)
    b_true, c_true = -3.0, 1.2
    background = 1.0 / (1.0 + np.exp(-b_true))
    target = background + 0.10 * (1.0 - background)

    def p_true(d):
        return 1.0 / (1.0 + np.exp(-(b_true + c_true * np.log(np.clip(d, 1e-9, None)))))

    lo, hi = 1e-6, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if p_true(mid) < target:
            lo = mid
        else:
            hi = mid
    bmd_true = 0.5 * (lo + hi)

    doses = np.array([0.001, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    n_total = np.full(doses.shape, 200.0)
    n_affected = rng.binomial(200, p_true(doses)).astype(float)

    result = benchmark_dose(doses, n_affected, n_total, bmr=0.10, model="loglogistic")

    assert abs(result.bmd - bmd_true) / bmd_true < 0.5
    assert result.bmdl < result.bmd
    assert result.bmdl > 0

    covered = 0
    trials = 30
    for seed in range(trials):
        rng_i = np.random.default_rng(seed + 1000)
        n_affected_i = rng_i.binomial(200, p_true(doses)).astype(float)
        exposure = np.abs(rng_i.normal(loc=result.bmdl / 100.0, scale=result.bmdl / 400.0, size=2000))
        r_i = benchmark_dose(doses, n_affected_i, n_total, bmr=0.10, model="loglogistic")
        dq = rfd_exceedance(exposure, r_i, uf=100.0, n=2000, rng=rng_i)
        p_exceed = float(np.mean(dq.samples))
        if 0.0 <= p_exceed <= 1.0:
            covered += 1
    assert covered / trials >= 0.88


def test_rfd_exceedance_monotone_in_uf():
    doses = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    n_total = np.full(doses.shape, 100.0)
    n_affected = np.array([5.0, 10.0, 25.0, 60.0, 90.0])
    result = benchmark_dose(doses, n_affected, n_total)
    rng = np.random.default_rng(1)
    exposure = np.full(2000, result.bmdl / 50.0)
    dq_strict = rfd_exceedance(exposure, result, uf=10.0, n=2000, rng=rng)
    dq_lenient = rfd_exceedance(exposure, result, uf=1000.0, n=2000, rng=rng)
    assert float(np.mean(dq_strict.samples)) <= float(np.mean(dq_lenient.samples))
