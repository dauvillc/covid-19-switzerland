"""
Clément Dauvilliers - EPFL TRANSP-OR lab semester project
09/03/2022

Tests the proper functioning of the AgeGroupSIR class.
"""
import numpy as np
from model import AgeGroupsSIR


def main():
    # ============= PARAMETERS ============== #
    # Age groups
    age_groups = [19, 64]

    # Force of infection for each age group
    lambdas = np.array([0.5, 0.5, 0.5])
    # Recovery rate for each age group
    gammas = np.array([0.2, 0.2, 0.2])

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
                          'lambdas': lambdas,
                          'gammas': gammas})

    # ============= SOLVING ================= #
    solved_states = model.solve(30)
    return 0


if __name__ == "__main__":
    main()