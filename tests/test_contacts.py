"""
Tests the functions implemented in contacts.py
"""
from model.contacts import load_population_contacts_csv


_CSV_PATH_ = "data/contact_counts.csv"


def main():
    types, ids, contact_matrices = load_population_contacts_csv(_CSV_PATH_)
    print(f"Loaded {contact_matrices.shape[0]} matrices of shape {contact_matrices[0].shape}")
    print("Found activity types: ", types)


if __name__ == "__main__":
    main()
