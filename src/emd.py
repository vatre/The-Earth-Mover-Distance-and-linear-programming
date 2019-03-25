import numpy as np
from scipy.optimize import linprog


def earth_mover_distance(p_values, q_values, p_weights=None, q_weights=None):
    # Checking constistency of lengths if weights are set
    if p_weights is not None and len(p_values) != len(p_weights):
        raise ValueError("p_value and p_weights should be of the same length."
                         "They are of lengths {len(p_values)} and {len(p_weights} respectively.")
    if q_weights is not None and len(q_values) != len(q_weights):
        raise ValueError("q_value and q_weights should be of the same length."
                         "They are of lengths {len(q_values)} and {len(q_weights} respectively.")

    # Setting default value for weights if they are not set
    if p_weights is None:
        p_weights = [1 / len(p_values)] * len(p_values)
    if q_weights is None:
        q_weights = [1 / len(q_values)] * len(q_values)

    # Setting variables
    min_total_weight = min(np.sum(p_weights), np.sum(q_weights))
    m, n = len(p_values), len(q_values)

    # Inequality constraints
    k = m * n
    A_ub = np.zeros((m + n, k))
    b_ub = np.zeros(m + n)
    # Total weight limit in outgoing flux from i
    for i in range(m):
        b_ub[i] = p_weights[i]
        A_ub[i, i * n:i * n + n] = np.ones(n)
    # Total weight limit in flux going into j
    for j in range(n):
        b_ub[m + j] = q_weights[j]
        for i in range(m):
            A_ub[m + j, i * n + j] = 1

    # Equality constraint
    A_eq = np.ones((1, k))
    b_eq = np.array([min_total_weight]).reshape(1, 1)

    # Build distance vector
    d = np.zeros(k)
    for i in range(m):
        for j in range(n):
            d[n * i + j] = np.linalg.norm(p_values[i] - q_values[j])

    res = linprog(d, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, options={"maxiter":500})
    distance = d @ res.x / min_total_weight

    # Bounds are by default non-negative so no need to change anything there
    return res, distance
