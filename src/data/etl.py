# ETL (Extract/Transform/Load) class
# loads a dataset from file and store it as a serialized object
# ------------------------------------------------------------------------
# Author: Moacir A. Ponti
# 2024
# ------------------------------------------------------------------------

import pickle
import pandas as pd


class ETL:

    def __init__(self):
        """
        Initializes the ETL class
        """
        self.dataset = None
        self.selected_features = None
        self.version = None

    def load_dataset_from_parquet(self, parquet_file):
        """
        Args:
            parquet_file: str - path to the parquet file containing the data to be inserted in the table
        Returns:
            list: the dataset read from the file
        """
        # opens csv file and reads the data, throws error if file is not found or not a csv file
        try:
            self.dataset = pd.read_parquet(parquet_file)
        except FileNotFoundError:
            raise FileNotFoundError(f'File {parquet_file} not found')
        except Exception:
            raise Exception(f'File {parquet_file} is not a valid file')

        return self.dataset

    def load_dataset_from_csv(self, csv_file):
        """
        Args:
            csv_file: str - path to the csv file containing the data to be inserted in the table
        Returns:
            list: the dataset read from the csv file
        """
        # opens csv file and reads the data, throws error if file is not found or not a csv file
        try:
            with open(csv_file, 'r') as file:
                csv_dataset = file.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f'File {csv_file} not found')
        except Exception:
            raise Exception(f'File {csv_file} is not a valid csv file')

        # # convert csv dataset into pandas dataframe
        self.dataset = pd.read_csv(csv_file)

        return self.dataset

    def select_columns(self, columns):
        """
        Args:
            columns: list - list of columns to be selected
        Returns:
            pd.DataFrame: the dataset with the selected columns
        """
        # select the columns from the dataset
        self.selected_features = self.dataset[columns].columns

        return self.selected_features

    def impute_missing_values(self, strategy='zero'):
        """
        Args:
            strategy: str - the strategy to impute the missing values
        Returns:
            pd.DataFrame: the dataset with the missing values imputed
        """
        if strategy == 'mean':
            self.dataset.fillna(self.dataset.mean(), inplace=True)
        elif strategy == 'zero':
            self.dataset.fillna(0.0, inplace=True)

    def serialize_dataset(self, dataset_artifact):
        """serializes the dataset into a pickle file
        Args:
            dataset_artifact: dict - the dataset to be serialized containing the name, version and the dataset
            output_file: str - the path to the output file
        """
        pickle.dump(dataset_artifact, open(f"dataset_{dataset_artifact['name']}_{dataset_artifact['version']}.pkl", 'wb'))

    @staticmethod
    def deserialize_dataset(input_file):
        """deserializes the dataset from a pickle file
        Args:
            input_file: str - the path to the input file
        Returns:
            pd.DataFrame: the deserialized dataset
        """
        return pd.read_pickle(input_file)

    @staticmethod
    def validate_dataset(dataset_artifact, check_missing=False):
        """validates the dataset
            - ensures that there are no missing values
            - ensures there exist the columns "event_timestamp", "target"
            - ensures there exist at least one more column other then the previous ones
        Args:
            dataset_artifact: dictionary containing: name, version and a pd.DataFrame
            check_missing: bool - whether to check for missing values
        Returns:
            bool: True if the dataset is valid, False otherwise
        """

        # check for
        if 'name' not in dataset_artifact or 'version' not in dataset_artifact:
            raise KeyError('Dataset artifact is not valid or missing name or version')

        # check for missing values
        if check_missing and dataset_artifact['dataset'].isnull().sum().sum() > 0:
            raise ValueError('Dataset contains missing values')

        # check for the column "target"
        if not all(col in dataset_artifact['dataset'].columns for col in ["target"]):
            raise ValueError('Dataset does not contain the column "target"')

        # check for at least one more column other then the previous ones
        if len(dataset_artifact['dataset'].columns) < 4:
            return False

        return True
