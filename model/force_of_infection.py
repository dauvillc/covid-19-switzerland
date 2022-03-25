"""
Clément Dauvilliers - Grenoble INP / EPFL Transport lab
23/03/2022

Defines functions used to compute the [aggregated] force of infection.
"""
import numpy as np


def compute_aggregated_new_infections(contact_matrices, beta):
    """
    From the contact matrices and the probabilities of transmission,
    computes the aggregated new infections (number of susceptible that an
    infectious individual contaminates per unit of time).
    :param contact_matrices: 3D int array of shape
        (sample_size, n_age_groups, n_activity_types).
    :param beta: float, probability of transmission.
    :return: the aggregated dI as an array of shape (n_age_groups).
    """
    return np.mean(contact_matrices, axis=(0, 2)) * beta
