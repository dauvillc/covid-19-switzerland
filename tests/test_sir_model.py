"""
Clément Dauvilliers - EPFL TRANSP-OR lab semester project
09/03/2022

Tests the proper functioning of the AgeGroupSIR class.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from model import AgeGroupsSIR


def main():
    # ============= PARAMETERS ============== #
    # Age groups
    age_groups = [9, 19, 35, 50, 64]
    n_age_groups = len(age_groups) + 1

    # Recovery rate for each age group
    gammas = np.array([0.2 for _ in range(n_age_groups)])

    # Contact matrices CSV file
    contacts_csv = "data/contact_counts.csv"

    # Initial state
    total_pop = 1000000
    age_pops = np.array([0.33 * total_pop] * n_age_groups)
    state0 = np.zeros((n_age_groups, 3))
    state0[:, 1] = np.array([100 for _ in range(n_age_groups)])  # initial infections
    state0[:, 0] = age_pops - state0[:, 1]
    rng = np.random.default_rng(seed=42)

    def initial_state_func():
        variation = rng.random((n_age_groups, 1)) * 0.4 + 0.8
        return variation * state0

    # ============= MODEL =================== #
    model = AgeGroupsSIR({'age_groups': age_groups,
                          'N': total_pop,
                          'gammas': gammas})
    model.load_contacts(contacts_csv)

    # ============= SOLVING ================= #
    test_probas = np.geomspace(1e-3, 1, 5)
    values = [test_probas for _ in range(6)]
    real_data = np.load(open('data/example_sim.npy', 'rb'))
    print(model.calibrate(real_data, values, 148, state0, day_eval_freq=1))
    _ = model.solve(148, initial_state_func, day_eval_freq=4, runs=10)
    fig = model.dashboard()
    fig.show()
    return 0


if __name__ == "__main__":
    main()