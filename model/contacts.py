"""
Clémetn Dauvilliers - Grenoble INP / EPFL Transport lab
23/03/2022

Defines functions linked to contact matrices and their aggregation.
"""
import pandas as pd
import numpy as np


def load_population_contacts_csv(csv_path):
    """
    Loads the contact matrices for a population, saved as a dataframe
    under the CSV format.
    :param csv_path: path to the CSV file containing the contacts for each
        individual. Each row should indicate the person ID, as well as the number
        of daily contacts of that person for each activity type and each age group.
    :return: (ids, matrices) where:
        - ids is an array giving the ID of every person in the dataset;
        - matrices is a 3D array of shape (n_people, n_age_groups, n_activity_types)
          such that matrices[i] is the contact matrix for person ids[i];
    """
    contacts_df = pd.read_csv(csv_path)
    # Isolates the value columns from the ids
    ids = contacts_df.index.to_numpy(dtype=np.int64)
    contacts_df = contacts_df.drop('id', axis=1)
    # Converts the columns (id excluded) into a 2D array
    matrices = contacts_df.to_numpy()

    # Computes the number of age groups and act. types from the columns' names
    # the column names are supposed to be 'activitytype_agegroup'
    n_age_groups = len(np.unique([col_name.split('_')[1] for col_name in contacts_df.columns]))
    n_activity_types = contacts_df.columns.shape[0] // n_age_groups
    # Reshapes into the expected 3D array
    matrices = matrices.reshape((matrices.shape[0], n_age_groups, n_activity_types), order='F')

    return ids, matrices
