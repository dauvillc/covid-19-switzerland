"""
Clément Dauvilliers - EPFL - TRANSP-OR Lab semester project
09/03/2022

Defines the AgeGroupsSIR class.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import product
from copy import deepcopy
from scipy.integrate import solve_ivp
from matplotlib import gridspec
from tqdm.auto import tqdm
from .equations import age_groups_SIR_derivative, state_as_vector
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
                                [f"{a + 1}-{b}" for a, b in zip(age_intervals, age_intervals[1:])] + \
                                [f'over {age_intervals[-1]}']

        # Attributes related to the computation of the new infections I*
        self.act_types_indices, self.sample_ids, self.contact_matrices_ = None, None, None
        self.avg_contact_matrix_, self.betas_, self.new_infections_ = None, None, None

        # Attributes filled after the model has been run
        self.solved_states_ = {}
        # Timepoints (in days) of the simulation, and timestamps corresponding to exact days
        # for example, [0.00, 0.25, 0.5, 0.75, 1] and [0.00, 1]
        self.timepoints_, self.days_, self.day_indices_ = [], [], []
        # Results of the equations system solver
        self.results_ = None
        # Number of new cases by the end of each day
        self.new_cases_ = None

    def load_contacts(self, contact_matrices_csv):
        """
        Loads the individual contact matrices from a CSV file and
        computes and saves their average.
        :param contact_matrices_csv: path to the CSV file containing the
            contact matrices.
        """
        self.act_types_indices, self.sample_ids, self.contact_matrices_ = load_population_contacts_csv(
            contact_matrices_csv)
        self.avg_contact_matrix_ = np.mean(self.contact_matrices_, axis=0)

    def set_betas(self, values):
        """
        Sets the probabilities of transmission of the model, then recomputes
        the force of infection with the new values.
        :param values: dict {activity type: proba}.
        """
        # Compiles the probabilities of transmission into an array, and makes sure
        # that the order corresponds to that of the contact matrices
        betas_array = np.empty((len(self.act_types_indices)), dtype=np.float64)
        for act_type, index in self.act_types_indices.items():
            betas_array[index] = values[act_type]

        # Saves the probabilities of transmission to plot them later
        # This dict is in the same order as the contact matrices' columns
        self.betas_ = pd.Series(data=betas_array, index=list(self.act_types_indices.keys()))

        # (Re)computes the new infections based on the probabilities of transmission
        if self.contact_matrices_ is None:
            raise ValueError("Please call load_contacts() before set_betas()")

        # Computes the new infections vector I* from the average contact matrix and the
        # probabilities of transmission.
        # This vector gives for each age group the daily new infections per Infectious individual
        # in the population.
        self.new_infections_ = np.sum(self.avg_contact_matrix_ * betas_array, axis=1)

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
        # Constructs evenly spaced timepoints to obtain
        step = 1 / day_eval_freq
        self.timepoints_ = np.arange(0, max_t + step, step)
        # Indices of timepoints that correspond to exact days
        self.day_indices_ = np.array([i for i in range(len(self.timepoints_))
                                      if i % day_eval_freq == 0])
        self.days_ = self.timepoints_[self.day_indices_]

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

        # Computes the NEW cases daily
        # We'll do it run by run, then concatenate all runs together
        runs_new_cases = []
        for run in range(runs):
            # Number of susceptible over time in this run
            susceptible = self.solved_states_[run][:, :, 0]
            # Filters to get only the susceptible at the END of each day, since
            # the equations are solved multiple times per day
            susceptible = susceptible[self.day_indices_]
            # New cases = Number of Susceptible on previous day - Number of susceptible
            new_cases = pd.DataFrame(data=-np.diff(susceptible, axis=0),
                                     index=self.days_[1:],
                                     columns=self.age_groups_names)
            new_cases['run'] = run
            new_cases['day'] = self.days_[1:]
            runs_new_cases.append(new_cases)
        self.new_cases_ = pd.concat(runs_new_cases)

        return deepcopy(self.solved_states_)

    def calibrate(self, real_data, test_values, max_t, initial_state, day_eval_freq=1,
                  verbose=True):
        """
        Optimizes the probability of transmission at every activity type to minimize the
        Mean Squared Error between the predicted and real trajectories.
        This is done in two steps:
        -- Optimize the new infections vector I*:
            The trajectory depend directly on the vector I* via the SIR equations. The optimal
            value for I* (whose length is the number of age groups) is found via grid search.
        -- Optimize the probabilities beta_a:
            The new infections vector I* depends on the contact matrix (which is fixed) and on the
            probabilities of transmission at each activity type. The second step tries to find the
            values for those probabilities which result in I* closest to its optimal value found in
            the previous step.

        :param real_data: dataframe/array of shape (simulation duration, number of age groups) giving
            the target trajectory for each age group.
        :param test_values: dictionary {activity type: values} where values is a list of probabilities
            to test for the corresponding activity type.
        :param max_t: duration of the simulation, in days, last included.
        :param initial_state: array of shape (n_age_groups, 3) giving the initial (S, I, R) state.
        :param day_eval_freq: integer; how many times the model should be evaluated per day.
        :param verbose: boolean, whether to print the current advancement
        :return: a dict {activity type: probability} giving the optimal betas. The model is also
            run with those values before returning.
        """
        if self.contact_matrices_ is None:
            raise ValueError('Please call load_contacts() before calibrating')
        peak_day = np.argmax(np.sum(real_data, axis=1))
        real_data = real_data[:peak_day]

        # FIRST STEP: Optimize the new infections vector I*
        # Puts mock values in the new infections
        self.new_infections_ = np.array([0.1 for _ in range(self.n_age_groups)])

        def evaluate_new_inf():
            """
            Returns the MSE between the predicted and real trajectories.
            """
            self.solve(max_t, lambda: initial_state, day_eval_freq)
            predicted = np.array(self.new_cases_.iloc[:, :-2])
            predicted = predicted[:peak_day]
            # Computes the mean squared error for each age group
            error = np.sum((predicted - real_data) ** 2, axis=0)
            # We want to fit all age groups, but their raw MSE aren't comparable
            # since the scale of the curves differ. We need to normalize the errors
            # to make them comparable:
            error = error / np.sum(real_data ** 2, axis=0)
            return np.sum(error)

        # Values for each age group to be tested for the grid search
        test_values = [np.linspace(0.01, 1, 20) for _ in range(self.n_age_groups)]
        total_tests = np.prod([len(values) for values in test_values])

        min_error, optimal_new_inf = float("+inf"), None
        for new_inf in tqdm(product(*test_values), total=total_tests):
            # Sets the value of the new infections for the current age group to the
            # value being tested
            self.new_infections_ = np.array(new_inf)
            error = evaluate_new_inf()
            if error < min_error:
                optimal_new_inf = self.new_infections_
                min_error = error

        self.new_infections_ = optimal_new_inf

        return optimal_new_inf

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
        sns.lineplot(data=results_long, x="day", y="infectious", hue="age group", style="age group", ax=ax)

        ax.set_ylabel("Infectious individuals")
        ax.set_xlabel("Days")
        ax.set_title("Infection trajectory")
        return ax

    def plot_betas(self, ax=None):
        """
        Plots the probabilities of transmission.
        :param ax: optional matplotlib axes to use;
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        # Bar plot of the probabilities of transmission
        sns.barplot(x=self.betas_.index, y=self.betas_.values, ax=ax)
        for idx, proba in enumerate(self.betas_):
            ax.text(idx, proba, str(proba), horizontalalignment="center")

        ax.set_xlabel("Activity type")
        ax.set_ylabel("Prob. of transmission")
        ax.set_ylim([0, self.betas_.values.max() * 1.2])
        ax.set_title('Activity-wise probabilities of transmission')
        return ax

    def plot_secondary_infections(self, ax=None):
        """
        Plots the average secondary infections per age group and decomposes
        it according to the weight of each activity type.
        :param ax: optional matplotlib to use;
        """
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        # Bar plot of the secondary infections
        sec_inf_weights = pd.DataFrame(data={act_type: self.avg_contact_matrix_[:, i]
                                             for i, act_type in enumerate(self.betas_.keys())},
                                       index=self.age_groups_names)
        sec_inf_weights.plot.bar(stacked=True, ax=ax)

        for idx, value in enumerate(self.new_infections_):
            ax.text(idx, value, f"{value: 1.2f}", horizontalalignment="center")

        ax.set_xlabel("Age group")
        ax.set_ylabel("Average daily secondary infections")
        ax.set_title("Activity weights in new infections rates")
        return ax

    def dashboard(self):
        """
        Creates a dashboard plotting various information that summarizes the model
        and its results.
        :return: a matplotlib Figure object.
        """
        fig = plt.figure(figsize=(11, 8))
        fig.suptitle("SIR Model Dashboard")
        gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[3, 1], width_ratios=[3, 2])
        self.plot_infections(ax=plt.subplot(gs[0, 0]))
        self.plot_betas(ax=plt.subplot(gs[1, 0]))
        self.plot_secondary_infections(ax=plt.subplot(gs[:, 1]))
        fig.tight_layout()
        return fig

    def plot_fit(self, real_data):
        """
        Plots together the real and predicted trajectories.
        :param real_data: ndarray of shape (duration_days, n_age_groups) giving the number
            of infections for each day and each age group.
        """
        if self.results_ is None:
            raise ValueError("Please call model.solve() before plotting")

        fig, axes = plt.subplots(nrows=self.n_age_groups, ncols=1, figsize=(10, 3 * self.n_age_groups))
        fig.suptitle('Predicted vs. real infections')

        # Converts the model's resuts (new cases trajectory) into long format
        # columns become: index day age_group infectious
        # multiple values may appear for the same age group and day
        # if several model runs were performed
        results_long = self.new_cases_.melt(id_vars='day',
                                            value_vars=self.age_groups_names,
                                            var_name="age group",
                                            value_name="new infections")

        # For each age group, plots the predicted and real trajectory
        for age_group_index, age_group in enumerate(self.age_groups_names):
            ax = axes[age_group_index]
            # Selects the results for the current age group
            age_group_results = results_long[results_long['age group'] == age_group]
            age_group_results = age_group_results.drop('age group', axis=1)
            real_traj = real_data[:, age_group_index]

            # Adds the real traj to the dataframe so that seaborn plots both trajectories automatically
            # and adjusts the style and legend.
            real_traj = pd.DataFrame({"day": self.days_[1:], "new infections": real_traj})
            # To do that, we need a new column which indicates whether the row is part of the real or
            # predicted trajectory
            age_group_results.loc[:, 'Trajectory'] = 'Predicted'
            real_traj.loc[:, 'Trajectory'] = 'Real'
            # We can now concatenate the dataframes:
            age_group_results = pd.concat([age_group_results, real_traj]).reset_index()

            # Plots the number of infectious per day per age group, and shows
            # the confidence interval
            sns.lineplot(data=age_group_results, x="day", y="new infections", hue="Trajectory", style="Trajectory",
                         ax=ax)

            ax.set_ylabel("Cases")
            ax.set_xlabel("Days")
            ax.set_title(f"Age group {age_group}")

        fig.tight_layout()
        return fig
