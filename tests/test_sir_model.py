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

    # activity-dependent probabilities of transmission
    betas = {
        "work": 0.03,
        "education": 0.1,
        "leisure": 0.01,
        "service": 0.05,
        "home": 0.05,
        "shop": 0.03
    }

    def initial_state_func():
        variation = rng.random(3) * 0.4 + 0.8
        return variation * state0

    # ============= MODEL =================== #
    model = AgeGroupsSIR({'age_groups': age_groups,
                          'N': total_pop,
                          'gammas': gammas})
    model.load_force_of_infection(contacts_csv, betas)

    # ============= SOLVING ================= #
    model.solve(60, initial_state_func, day_eval_freq=4, runs=10)
    ## fig = model.dashboard()
    fig = model.plot_fit(np.arange(60 * 3).reshape((3, 60)))
    fig.savefig("figures/example_run.png")
    fig.show()
    return 0


if __name__ == "__main__":
    main()