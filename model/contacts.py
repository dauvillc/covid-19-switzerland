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
    :return: (activity_types, ids, matrices) where:
        - activity_types is a dict {type: index} where type is a type of activity (eg
            'leisure' and index is the index of that type in the contact matrices'
            columns.
        - ids is an array giving the ID of every person in the dataset;
        - matrices is a 3D array of shape (n_people, n_age_groups, n_activity_types)
          such that matrices[i] is the contact matrix for person ids[i];
    """
    contacts_df = pd.read_csv(csv_path, index_col=0)
    # Isolates the value columns from the ids
    ids = contacts_df.index.to_numpy(dtype=np.int64)
    # Converts the columns (id excluded) into a 2D array
    matrices = contacts_df.to_numpy()

    # Computes the number of age groups and act. types from the columns' names
    # the column names are supposed to be 'activitytype_agegroup'
    split_column_names = [col_name.split('_') for col_name in contacts_df.columns]
    activity_types = np.unique([act_type for act_type, _ in split_column_names])
    age_groups = np.unique([age_group for _, age_group in split_column_names])

    # Stores the index of each activity type in the contact matrices' columns
    act_types_indices = {type_: idx for idx, type_ in enumerate(activity_types)}

    # Reshapes into the expected 3D array
    n_age_groups = len(age_groups)
    n_activity_types = len(activity_types)
    matrices = matrices.reshape((matrices.shape[0], n_age_groups, n_activity_types), order='F')

    return act_types_indices, ids, matrices
