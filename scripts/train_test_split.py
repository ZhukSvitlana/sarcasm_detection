import pandas as pd
import os
from sklearn.model_selection import train_test_split

def make_split(path_to_csv, path_to_save, test_size):
    data = pd.read_csv(path_to_csv)

    X_train, X_test = train_test_split(data, test_size=test_size)

    X_train.to_csv(os.path.join(path_to_save, 'train.csv'))
    X_test.to_csv(os.path.join(path_to_save, 'test.csv'))

make_split.__doc__="""
Splits the dataset into training and testing sets and saves them as CSV files.

    Parameters:
        path_to_csv (str): Path to the CSV file containing the dataset.
        path_to_save (str): Path to the directory where the split datasets will be saved.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        None
        """

# Using Fire to create a command-line interface for the make_split function.
if __name__ == "__main__":
    from fire import Fire

    Fire(make_split)