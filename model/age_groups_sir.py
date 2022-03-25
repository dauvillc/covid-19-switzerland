"""
Clément Dauvilliers - EPFL - TRANSP-OR Lab semester project
09/03/2022

Defines the AgeGroupsSIR class.
"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.integrate import solve_ivp
from .equations import age_groups_SIR_derivative, state_as_vector, state_as_matrix
from .force_of_infection import compute_aggregated_new_infections
from .contacts import load_population_contacts_csv


class AgeGroupsSIR:
    """
    Defines the SIR model class used to represent the successive waves of COVID-19 in Switzerland,
    based on socio-economic and demographic variables.
    The SIR model is split into three agre groups: 0-19, 20-64, over 65
    """
    def __init__(self, parameters):
        """
        Creates an age-groups SIR model based on given parameters.
        :param parameters: dict-like object giving the value of each parameter.
        Parameters are:
        - 'age_groups': list/array of integers giving the boundaries of each age group.
                        For example, [19, 64] indicates 0-19, 20-64 and over 65.
        - 'N': population size;
        - 'initial_state': list of length k where k is the number of age groups.
                           Each entry in the list is an array of length 2 giving the initial
                           number of people infected and recovered within the corresponding
                           age group. For example, [[189, 1, 0], [0, 0, 0], [9, 0, 1]] means that initially,
                           1 person below 19 y.o. is infected, and that a single person over 65
                           has already recovered. The total population is 200.
        - 'gammas': array/list of length (number of age groups,) giving the recovery rates for each age
                    group.
        """
        # We copy the dict to be safe from external modifications
        self.params_ = deepcopy(parameters)

        # Save the number of age groups to access it easily later
        self.n_age_groups = len(self.params_['age_groups']) + 1

        # Create the state matrix: it has dimensions Na x 3, and counts the number of people in each
        # SIR state for each age group. Each row corresponds to an age group.
        self.initial_state_ = np.zeros((self.n_age_groups, 3))
        # Fill that matrix with the initial state
        for group in range(self.n_age_groups):
            self.initial_state_[group] = self.params_['initial_state'][group]

        self.sample_ids, self.contact_matrices_ = None, None
        self.new_infections_ = None
        self.solved_states_ = None

    def load_force_of_infection(self, contact_matrices_csv, betas):
        """
        Computes the age-wise force of infection.
        :param contact_matrices_csv: path to the CSV file containing the
            contact matrices.
        :param betas: vector of length n_age_groups, giving the probability of
        infection for each age group.
        """
        self.sample_ids, self.contact_matrices_ = load_population_contacts_csv(contact_matrices_csv)
        self.new_infections_ = compute_aggregated_new_infections(self.contact_matrices_, betas)

    def solve(self, max_t):
        """
        Solves the ODE system with an increment of 1 day from t=0 to max_t (excluded).
        :param max_t: last day of the simulation.
        :return: an array of shape (max_t, number of age groups, 3) giving the state of the
            system at each time.
        """
        # Builds the equation function from the parameters
        eq_func = age_groups_SIR_derivative(self.new_infections_, self.params_['gammas'])
        # Converts the initial state to a vector to match the solver's signature
        initial_state = state_as_vector(self.initial_state_)
        # Solves the ODE numerically
        solution = solve_ivp(eq_func,
                             y0=state_as_vector(initial_state),
                             t_span=(0, max_t),
                             t_eval=np.arange(0, max_t),
                             vectorized=True)
        solved_states = solution.y.T
        # Reshape the states back to matrices and return them
        self.solved_states_ = solved_states.reshape((solved_states.shape[0], 3, 3))
        return np.copy(self.solved_states_)

    def plot_infections(self, max_t, ax=None):
        """
        Solves the model and plots the age-wise incidence curve.
        :param max_t: last day of the simulation.
        :param ax: matplotlib ax to use for plotting
        """
        # If no ax was provided, create the figure
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        # If the model hasn't been solved yet, solve it now
        if self.solved_states_ is None:
            self.solve(max_t)

        infection_states = self.solved_states_[:, :, 1]
        days = np.arange(max_t) + 1
        ax.plot(days, infection_states[:, 0])
        ax.plot(days, infection_states[:, 1])
        ax.plot(days, infection_states[:, 2])

        age_intervals = self.params_['age_groups']
        ax.legend([f'$\leq$ {age_intervals[0]} y.o.'] +\
                  [f"{a}-{b}" for a, b in zip(age_intervals, age_intervals[1:])] +\
                  [f'$>$ {age_intervals[-1]}'])
        return fig, ax
