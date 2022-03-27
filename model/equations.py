"""
Clément Dauvilliers - EPFL - TRANSP-OR Lab semester project
09/03/2022

Defines the functions associated with the ODE system of the SIR model.
"""

import numpy as np


def state_as_matrix(state):
    """
    Returns a SIR state viewed as a matrix.
    :param state: array of shape (number of age groups x 3,)
    :return: a matrix of shape (number of age groups, 3).
        Each row gives the number of Susceptible, Infectious and Recovered
        individuals.
    """
    return state.reshape((-1, 3))


def state_as_vector(state):
    """
    Returns a SIR state viewed as a vector.
    :param state: matrix of shape (number of age groups, 3)
    :return: a vector of length (number of age groups x 3)
    """
    return state.reshape((-1))


def age_groups_SIR_derivative(lambdas, gammas):
    """
        :param lambdas: array of length <number of age groups> giving the force of infection
            for each age group.
        :param gammas: array of length <number of age groups> giving the recovery rate
            for each age group.
        :return: the derivative of the state at the given time step.
    """

    def derivative(time, state):
        """
        Function f so that dY/dt = f(Y) is the ODE system of the SIR model
        split into age groups.
        :param time: time step (in day).
        :param state: vector of length (number of age groups * 3). Should be an age-group SIR matrix
            vectorized via state_as_matrix.
        :return: the value of dS/dt on day t as a vector of same length as state.
        """
        # We convert the state back into matrix form to retrieve the SIR
        # values
        state = state_as_matrix(state)
        S, I, R = state[:, 0], state[:, 1], state[:, 2]
        N = np.sum(state)

        # Implements the ODE system
        dS = - lambdas.T * I * (S / N)
        dR = gammas * I
        dI = - dS - dR

        # Transforms the state back into a vector for coherence with the
        # usual python ODE solvers (such as scipy.integrate.odeint).
        return state_as_vector(np.stack([dS, dI, dR], axis=1))

    return derivative
