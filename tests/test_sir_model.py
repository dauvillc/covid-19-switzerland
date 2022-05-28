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
    age_groups = [19, 64]

    # Recovery rate for each age group
    gammas = np.array([0.2, 0.2, 0.2])

    # Contact matrices CSV file
    contacts_csv = "data/contact_counts.csv"

    # Initial state
    total_pop = 1000000
    # age_pops = np.array([0.2 * total_pop, 0.65 * total_pop, 0.15 * total_pop])
    age_pops = np.array([0.33 * total_pop] * 3)
    state0 = np.zeros((3, 3))
    state0[:, 1] = np.array([100, 100, 100])  # initial infections
    state0[:, 0] = age_pops - state0[:, 1]
    rng = np.random.default_rng(seed=42)

    def initial_state_func():
        variation = rng.random(3) * 0.4 + 0.8
        return variation * state0

    # ============= MODEL =================== #
    model = AgeGroupsSIR({'age_groups': age_groups,
                          'N': total_pop,
                          'gammas': gammas})
    model.load_contacts(contacts_csv)

    # ============= SOLVING ================= #
    test_probas = np.geomspace(1e-3, 1, 5)
    values = {
        "work": test_probas,
        "education": test_probas,
        "leisure": test_probas,
        "service": test_probas,
        "home": test_probas,
        "shop": test_probas
    }
    """
    betas = {
        "work": 0.2,
        "education": 0.1,
        "leisure": 0.1,
        "service": 0.1,
        "home": 0.1,
        "shop": 0.1
    }
    """
    real_data = np.load(open('data/example_sim.npy', 'rb'))
    print(model.calibrate(real_data, values, 148, state0, day_eval_freq=1))
    # model.set_betas(betas)
    _ = model.solve(148, initial_state_func, day_eval_freq=4, runs=10)
    fig = model.plot_fit(real_data)
    fig.show()
    return 0


if __name__ == "__main__":
    main()