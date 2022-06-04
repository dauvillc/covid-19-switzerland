"""
Clément Dauvilliers - EPFL - 01/05/2022

Implements the contacts prediction model
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV


def to_binary_nonzero(array):
    """
    Converts an array to binary values indicating nonzero entries.
    :param array: ndarray.
    :return: an array of identical shape whose values are 0 or 1 based on
        whether the original value is zero or nonzero.
    """
    return (array > 0).astype(int)


class ContactsPredictor:
    """
    Model that predicts the contact matrix for a individual based on their
    socio-economic factors.
    """
    def __init__(self, max_depth=10, verbose=True):
        """
        :param max_depth: max depth of the underlying Random Forest Regressor.
        :param verbose: boolean, whether to output detailed information about the
        various processes.
        """
        self.max_depth = max_depth
        self.verbose = verbose

    def fit(self, X, y):
        """
        :param X: pd Dataframe of shape (nb_samples, nb_variables)
        :param y: pd Dataframe of shape (nb_samples, predicted_vars),
                  flattened contact matrices.
        :return:  the fitted object (self).
        """
        self.features_names_ = X.columns
        self.predicted_variables_names_ = y.columns

        if self.verbose:
            print(f"Fitting input of shape {X.shape} to {y.shape[1]} features")
            print(f"Max depth: {self.max_depth}")

        # First step: predicts whether there is any contact
        binary_y = to_binary_nonzero(y)
        self.binary_models = dict()
        # Each column of Y is a predicted contact variable
        for contacts_category, target in binary_y.iteritems():
            if self.verbose:
                print(f"Fitting contacts category {contacts_category}")
            rfecv = RFECV(estimator=RandomForestClassifier(max_depth=self.max_depth, random_state=42),
                          step=4, cv=5).fit(X, target)
            self.binary_models[contacts_category] = rfecv

        return self

    def score(self, X, y):
        """
        :param X: pd Dataframe of shape (nb_samples, np_variables)
        :param y: pd Dataframe of shape (nb_samples, predicted_vars), flattened contact matrices.
        :return: (CS, RS) where:
            -- CS is the binary classification score, the accuracy at predicting whether an individual's
            contact matrix entry is zero;
            -- RS is the regression score: for an individual and a contact matrix entry that is known to be
            non zero, corresponds to the average R2 score.
        """
        # Evaluates the classification models
        binary_y = to_binary_nonzero(y)
        classification_scores = dict()
        for contact_category, target in binary_y.iteritems():
            rfecv = self.binary_models[contact_category]
            classification_scores[contact_category] = rfecv.score(X, target)

        return classification_scores
