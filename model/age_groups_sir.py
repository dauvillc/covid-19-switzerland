"""
Clément Dauvilliers - EPFL - TRANSP-OR Lab semester project
09/03/2022

Defines the AgeGroupsSIR class.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from copy import deepcopy
from scipy.integrate import solve_ivp
from .equations import age_groups_SIR_derivative, state_as_vector, state_as_matrix
from .force_of_infection import compute_aggregated_new_infections
from .contacts import load_population_contacts_csv

sns.set_theme()


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
        - 'gammas': array/list of length (number of age groups,) giving the recovery rates for each age
                    group.
        """
        # We copy the dict to be safe from external modifications
        self.params_ = deepcopy(parameters)

        # Save the number of age groups to access it easily later
        self.n_age_groups = len(self.params_['age_groups']) + 1
        age_intervals = self.params_['age_groups']
        self.age_groups_names = [f'under {age_intervals[0] + 1}'] + \
                                [f"{a}-{b}" for a, b in zip(age_intervals, age_intervals[1:])] + \
                                [f'over {age_intervals[-1]}']

        self.act_types_indices, self.sample_ids, self.contact_matrices_ = None, None, None
        self.new_infections_ = None
        self.solved_states_ = {}
        self.timepoints_ = []
        self.results_ = None

    def load_force_of_infection(self, contact_matrices_csv, betas):
        """
        Computes the age-wise force of infection.
        :param contact_matrices_csv: path to the CSV file containing the
            contact matrices.
        :param betas: dictionnary {activity type: p} where p is the probability
            of transmission associated with the activity type.
        """
        self.act_types_indices, self.sample_ids, self.contact_matrices_ = load_population_contacts_csv(
            contact_matrices_csv)

        # Compiles the probabilities of transmission into an array, and makes sure
        # that the order corresponds to that of the contact matrices
        betas_array = np.empty((len(self.act_types_indices)), dtype=np.float64)
        for act_type, index in self.act_types_indices.items():
            betas_array[index] = betas[act_type]

        self.new_infections_ = compute_aggregated_new_infections(self.contact_matrices_,
                                                                 betas_array)

    def solve(self, max_t, initial_state_func, day_eval_freq=1, runs=1):
        """
        Solves the ODE system with an increment of 1 day from t=0 to max_t (excluded).
        :param max_t: last day of the simulation.
        :param initial_state_func: function f(void) --> array such that a call to f returns
            an initial state, as a 2D matrix of dim (n_age_groups, 3).
            The function can always return the same state but may also include randomization
            to study the sensitivity of the model.
        :param day_eval_freq: integer; how many times the model should be evaluated per day.
        :param runs: how many simulations to perform. Multiple simulations allows to compute
            confidence intervals over the random parameters, such as initial state or probability
            of transmission.
        :return: an array of shape (max_t * day_eval_freq, number of age groups, 3) giving the state of the
            system at each time.
        """
        # Builds the equation function from the parameters
        eq_func = age_groups_SIR_derivative(self.new_infections_, self.params_['gammas'])
        self.timepoints_ = np.linspace(0, max_t, max_t * day_eval_freq)

        results = []
        for run in range(runs):
            # Converts the initial state to a vector to match the solver's signature
            initial_state = state_as_vector(initial_state_func())
            # Solves the ODE numerically
            solution = solve_ivp(eq_func,
                                 y0=state_as_vector(initial_state),
                                 t_span=(0, max_t),
                                 t_eval=self.timepoints_,
                                 vectorized=True)
            solved_states = solution.y.T
            # Reshape the states for this run back to matrices and saves them
            self.solved_states_[run] = solved_states.reshape((solved_states.shape[0], self.n_age_groups, 3))

            # Creates a pandas dataframe to store this run's results
            # Those results contain the number of infectious at each day, in each age group
            # self.solved_states_[run][:, i, 1] is the number of infectious for each time stamp, in the
            # current run's results
            data = {self.age_groups_names[i]: self.solved_states_[run][:, i, 1] for i in range(self.n_age_groups)}
            data['run'] = [run for _ in range(self.timepoints_.shape[0])]
            data['day'] = self.timepoints_
            results.append(pd.DataFrame(data=data,
                                        index=self.timepoints_))

        # Assembles the results of all runs into a single dataframe
        self.results_ = pd.concat(results)
        return np.copy(self.solved_states_)

    def plot_infections(self, ax=None):
        """
        For an already solved model, plots the age-wise incidence curve.
        :param ax: optional matplotlib ax to use for plotting
        """
        if self.results_ is None:
            raise ValueError("Please call model.solve() before plotting")
        # If no ax was provided, create the figure
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)

        # Converts the model results into long format
        # columns become: index day age_group infectious
        # multiple values may appear for the same age group and day
        # if several model runs were performed
        results_long = self.results_.melt(id_vars=['day'],
                                          value_vars=self.results_.columns[:-2],
                                          var_name="age group",
                                          value_name="infectious")
        # Plots the number of infectious per day per age group, and shows
        # the confidence interval
        sns.lineplot(data=results_long, x="day", y="infectious", hue="age group", ax=ax)

        ax.set_ylabel("Infectious individuals")
        ax.set_xlabel("Days")
        return ax

    def plot_secondary_infections(self, ax=None):
        """
        Makes a heatmap of the average number of secondary infections
        per age group and activity type.
        :param ax: optional matplotlib Axes object to use for plotting;
        """
        # TODO
        pass
