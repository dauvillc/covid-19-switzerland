"""
Clément Dauvilliers - Grenoble INP / EPFL Transport lab
23/03/2022

Defines functions used to compute the [aggregated] force of infection.
"""
import numpy as np


def compute_aggregated_new_infections(avg_contact_matrix, beta):
    """
    From the average contact matrix and the probabilities of transmission,
    computes the aggregated new infections I* (number of susceptible that an
    infectious individual contaminates per unit of time).
    :param avg_contact_matrix: 2D int array of shape
        (n_age_groups, n_activity_types).
    :param beta: vector of length n_activity_types giving the probability
        of transmission for each activity type.
    :return: the aggregated dI as an array of shape (n_age_groups).
    """
    avg_contact_matrix = avg_contact_matrix * beta
    return np.sum(avg_contact_matrix, axis=1)


def average_contact_matrix(contact_matrices, beta):
    """
    From the contact matrices and the probabilities of transmission,
    computes the average contact matrix.
    :param contact_matrices: 3D int array of shape
        (sample_size, n_age_groups, n_activity_types)
    :param beta: vector of length n_activity_types giving the
        probability of transmission for contact occurring in each
        type of activity.
    :return: the average contact matrix as an array of shape
        (n_age_groups, n_activity_types).
    """
    avg_contacts = np.mean(contact_matrices, axis=0)
    # Ponders the average contacts with the probs. of transmission
    return avg_contacts * beta

