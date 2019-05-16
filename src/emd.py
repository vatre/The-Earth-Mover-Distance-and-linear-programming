import numpy as np
from scipy.optimize import linprog


def build_LP(p_values, q_values, p_weights=None, q_weights=None):
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
    total_weight = (np.sum(p_weights), np.sum(q_weights))
    min_total_weight = min(total_weight)
    m, n = len(p_values), len(q_values)

    # Inequality constraints
    k = m * n
    A_ub = np.zeros((n + m, k)) # initial point total flux
    b_ub = np.concatenate((p_weights, q_weights))
    total_sent = np.array([1] * n + [0] * (k - n))

    total_received = [0] * k
    for i in range(m):
        total_received[i * n] = 1
    total_received = np.array(total_received)
    # Total weight limit in outgoing flux from i
    for i in range(m):
        A_ub[i] = np.roll(total_sent, n * i)
    for i in range(n):
        A_ub[m + i] = np.roll(total_received, i)

    # Equality constraint
    A_eq = np.ones((1, k)) # initial point total flux
    b_eq = np.array([min_total_weight])

    # Build distance vector
    d = np.zeros(k)
    k = 0
    for i in range(m):
        for j in range(n):
            d[k] = np.linalg.norm(p_values[i] - q_values[j])
            k += 1

    return {
      "c": d,
      "A_ub": A_ub,
      "b_ub": b_ub,
      "A_eq": A_eq,
      "b_eq": b_eq,
    }

def LP_2_MPS(LP, output_file="lp.mps"):
    lines = []
    lines.append("NAME          LPEMD\n")
    lines.append("ROWS\n")
    lines.append(" N  COST\n")
    n_ineq = LP["A_ub"].shape[0]
    n_eq = LP["A_eq"].shape[0]
    n_x = len(LP["c"])
    for i in range(n_ineq):
        lines.append(" L  LIM" + str(i) + "\n")
    for i in range(n_eq):
        lines.append(" E  EQ" + str(i) + "\n")
    lines.append("COLUMNS" + "\n")
    for i in range(n_x):
        var = "X" + str(i)
        for j in range(n_ineq):
            eq = "LIM" + str(j)
            value = LP["A_ub"][j][i]
            if value != 0:
                value = str(np.round(value, decimals=4))
                temp = " " * 4 + var
                temp += " " * (14 - len(temp)) + eq
                temp += " " * (36 - len(temp) - len(value)) + value
                lines.append(temp + "\n")
        for j in range(n_eq):
            eq = "EQ" + str(j)
            value = LP["A_eq"][j][i]
            if value != 0:
                value = str(np.round(value, decimals=4))
                temp = " " * 4 + var
                temp += " " * (14 - len(temp)) + eq
                temp += " " * (36 - len(temp) - len(value)) + value
                lines.append(temp + "\n")
        temp = " " * 4 + var
        temp += " " * (14 - len(temp)) + "COST"
        value = str(np.round(LP["c"][i], decimals=2))
        temp += " " * (36 - len(temp) - len(value)) + value
        lines.append(temp + "\n")
    lines.append("RHS" + "\n")
    for j in range(n_ineq):
        eq = "LIM" + str(j)
        value = LP["b_ub"][j]
        value = str(np.round(value, decimals=4))
        temp = "     RHS1     " + eq
        temp += " " * (36 - len(temp) - len(value)) + value
        lines.append(temp + "\n")
    for j in range(n_eq):
        eq = "EQ" + str(j)
        value = LP["b_eq"][j]
        value = str(np.round(value, decimals=4))
        temp = "     RHS1     " + eq
        temp += " " * (36 - len(temp) - len(value)) + value
        lines.append(temp + "\n")
    lines.append("ENDATA")
    f = open(output_file, "w")
    f.writelines(lines)
    f.close()


def kmeans_to_hw(y):
    n_classes = np.sort(np.unique(y))
    n_items = len(y)
    h1 = np.zeros((len(n_classes), n_items))
    w = np.zeros(len(n_classes))
    for c in n_classes:
        h = (y == c).astype(int)
        h1[c] = h
        w[c] = h.sum() / n_items
    return h1, w

def kmeans_to_dist(y1, cent1, y2, cent2):
    n_item = len(y1)
    n_classes_1 = len(np.sort(np.unique(y1)))
    n_items_1 = len(y1)
    n_classes_2 = len(np.sort(np.unique(y2)))
    n_items_2 = len(y2)

    m = n_classes_1
    n = n_classes_2
    d = np.zeros(m * n)
    k = 0
    for i in range(m):
        for j in range(n):
            for l in range(n_item):
                d1 = int(y1[l] == i)
                d2 = int(y2[l] == j)
                d[k] += np.abs(d1 - d2) # d1 * d2 * np.linalg.norm(cent1[i] - cent2[j])
            k += 1
    return d