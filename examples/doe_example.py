"""mixle.doe: design of experiments, Bayesian optimization, and sensitivity analysis.

Three staples on one objective: a space-filling Latin-hypercube design, Bayesian optimization that
minimizes a function in a handful of evaluations, and Sobol indices that attribute output variance
to each input.
"""

import numpy as np

from mixle.doe import latin_hypercube, minimize, sobol_indices

if __name__ == '__main__':
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]

    # 1. Space-filling design -- one well-spread point per row.
    design = latin_hypercube(bounds, 6, seed=0)
    print('Latin hypercube (6 x 2):')
    print(np.round(design, 2))

    # 2. Bayesian optimization of a quadratic whose minimum is at (1, -2).
    objective = lambda p: float((p[0] - 1.0) ** 2 + (p[1] + 2.0) ** 2)
    res = minimize(objective, bounds, n_init=5, n_iter=15, seed=0)
    print('\nBayesOpt best_x   true (1.0, -2.0)  fit %s' % np.round(res.best_x, 2))
    print('BayesOpt best_y   %.4f' % res.best_y)

    # 3. Sobol sensitivity of y = x0^2 + 0.5 * x1 -- x0 should dominate the variance.
    si = sobol_indices(lambda X: X[:, 0] ** 2 + 0.5 * X[:, 1], bounds, n=2048, names=['x0', 'x1'])
    clean = lambda v: {n: round(float(x), 3) for n, x in zip(si['names'], v)}
    print('\nSobol first-order S1:', clean(si['S1']))
    print('Sobol total-order  ST:', clean(si['ST']))
