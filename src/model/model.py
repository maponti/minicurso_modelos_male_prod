# Implements the model class
# Takes in the data using ETL class and trains a LightGBM model
# ------------------------------------------------------------------------
# Author: Moacir A. Ponti
# 2024
# ------------------------------------------------------------------------

import pickle
import json
import numpy as np
import lightgbm as lgb
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, version=None):
        """
        Initializes the Model class
        """
        self.name = None
        self.model = None
        self.version = version
        self.features = None
        self.metrics = None
        self.instance_example = None

    def create_train_test_split(self, dataset, test_size=0.2, split_by='time',
                                time_column=None, target_column='target',
                                ignore_columns=None):
        """creates the train test split
        Args:
            dataset: dict - the dataset artifact
            test_size: float - the test size
            split_by: str - the split method, 'time' (default) or 'random'
            time_column: str - the time column
            target_column: str - the target column
            ignore_columns: list - the columns to ignore
        Returns:
            tuple: the train test split with X, y, X_test, y_test
        """
        if ignore_columns is None:
            ignore_columns = []

        if dataset is None:
            raise ValueError('Could not load dataset from artifact')

        if split_by == 'time':
            if time_column is None:
                raise ValueError('Time column must be provided when split_by is time')

            # sort the dataframe dataset by time
            dataset['dataset'].sort_values(time_column, inplace=True)
            # create split column
            dataset['split'] = 'train'
            # set the test split using the test_size last rows
            dataset['dataset'].loc[dataset['dataset'].index[-int(test_size*len(dataset)):], 'split'] = 'test'

            idx_train = dataset['dataset']['split'] == 'train'
            idx_test = dataset['dataset']['split'] == 'test'

            X = dataset['dataset'].loc[idx_train].drop(columns=[target_column] + ignore_columns, inplace=False)
            y = dataset['dataset'].loc[idx_train, target_column].values
            X_test = dataset['dataset'].loc[idx_test].drop(columns=[target_column] + ignore_columns, inplace=False)
            y_test = dataset['dataset'].loc[idx_test, target_column].values

        elif split_by == 'random':
            # random split
             X, X_test, y, y_test = train_test_split(dataset['dataset'].drop(columns=[target_column] + ignore_columns, inplace=False),
                                                    dataset['dataset'][target_column].values, test_size=test_size)
        else:
            raise ValueError('Split method not recognized')

        self.features = X.columns
        self.instance_example = X_test.iloc[-1].to_dict()

        return X, X_test, y, y_test

    def train(self, X, y, model_params, random_state=42):
        """trains the model
        Args:
            model_params: dict - the model parameters
            random_state: int - the random state
        """
        assert len(X) == len(y)

        self.model_name = model_params['model_name']
        self.model = lgb.LGBMClassifier(**model_params['model_params'], random_state=random_state)
        self.model.fit(X, y)

    def predict(self, x_data):
        """predicts the target
        Args:
            x_data: np.array - the data to predict
        Returns:
            np.array: the predictions
        """
        n_feats = len(self.features)

        missing_features = 0
        data_array = np.zeros(n_feats)
        for i, feature in enumerate(self.features):
            if feature not in x_data:
                x_data[feature] = 0
                missing_features += 1
            data_array[i] = x_data[feature]

        score = self.model.predict_proba(data_array.reshape(1, -1))[0]

        return score, missing_features

    def validate_model(self):
        """validates the model was trained """
        if self.model is None:
            raise ValueError('Model has not been trained')
        if self.predict([self.features]) is None:
            raise ValueError('Model does not predict')
        return True

    def evaluate_model(self, X_test, y_test, dataset_version, threshold=0.5, verbose=False):
        """evaluates the model
        Args:
            X_test: np.array - the test data
            y_test: np.array - the test target
            dataset_version: str - the version of the dataset
            threshold: float - the threshold for the predictions
            verbose: bool - whether to print the evaluation metrics
        Returns:
            dict: the evaluation metrics containing prauc, rocauc and f1
        """

        y_pred = self.model.predict(X_test)
        prauc = average_precision_score(y_test, y_pred)
        rocauc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, (y_pred >= threshold).astype(int))

        if verbose:
            print(f'Model (v={self.version}) evaluation:')
            print(f'Test set version = {dataset_version}, size = {X_test.shape[0]} rows x {X_test.shape[1]} feats')
            print(f'\tPRAUC: {prauc:.4f}')
            print(f'\tROCAUC: {rocauc:.4f}')
            print(f'\tF1: {f1:.4f}')

        self.metrics = {'prauc': prauc, 'rocauc': rocauc, 'f1': f1}

        return self.metrics

    def instance_example_to_json(self, json_file='instance.json'):
        with open(json_file, 'w') as file:
            json.dump(self.instance_example, file)

    def serialize(self):
        """serializes the model into a pickle file"""
        pickle.dump(self, open(f'model_{self.version}.pkl', 'wb'))

    @staticmethod
    def load(serialized_model_file):
        """loads the model from a pickle file"""
        with open(serialized_model_file, 'rb') as file:
            return pickle.load(file)