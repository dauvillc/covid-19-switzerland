"""
Clément Dauvilliers - EPFL TRANSP-OR lab semester project
09/03/2022

Tests the proper functioning of the AgeGroupSIR class.
"""
import numpy as np
import matplotlib.pyplot as plt
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
    age_pops = np.array([0.2 * total_pop, 0.65 * total_pop, 0.15 * total_pop])
    state0 = np.zeros((3, 3))
    state0[:, 1] = np.array([237, 748, 45])  # initial infections
    state0[:, 0] = age_pops - state0[:, 1]

    # ============= MODEL =================== #
    model = AgeGroupsSIR({'age_groups': age_groups,
                          'N': total_pop,
                          'initial_state': state0,
                          'gammas': gammas})
    model.load_force_of_infection(contacts_csv, 0.2)

    # ============= SOLVING ================= #
    fig, ax = model.plot_infections(30)
    fig.show()
    return 0


if __name__ == "__main__":
    main()