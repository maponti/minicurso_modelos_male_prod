# Runs training from the command line
# ------------------------------------------------------------------------
# Author: Moacir A. Ponti
# 2024
# ------------------------------------------------------------------------

import sys
import os
import json
import pickle

from model import Model
from ...src.data.etl import ETL


def train_and_evaluate(dataset_artifact_file, model_version, model_params_file):
    """ Trains and Evaluate model
    Args:
        dataset_artifact_file: str - the path to the dataset artifact file
        model_version: str - the model version
        model_params_file: str - the path to the model parameters file
    Returns:
        tuple: model, dictionary with evaluation
    """
    # create an instance of the Model class
    model = Model(model_version)

    # loads dataset artifact from the pickle file
    dataset_artifact = ETL.deserialize_dataset(dataset_artifact_file)
    if not dataset_artifact:
        print('Could not load dataset artifact')
        return

    # validate dataset
    if not ETL.validate_dataset(dataset_artifact):
        print('Dataset is not valid')
        return

    # loads JSON into a dictionary
    with open(model_params_file, 'r') as file:
        model_params = json.load(file)

    # get columns from dataset and exclude those in dataset_artifact['features']
    ignore_columns = [col for col in dataset_artifact['dataset'].columns if col not in dataset_artifact['features']]

    X, y, X_test, y_test = model.create_train_test_split(dataset_artifact, test_size=0.2, split_by='random',
                                                         time_column=None, target_column='target',
                                                         ignore_columns=ignore_columns)
    model.train(X, y, model_params, text_column='item_title')
    dict_eval = model.evaluate_model(X_test, y_test, dataset_version=dataset_artifact['version'], verbose=True)

    return model, dict_eval


def main():
    # loads the dataset artifact from the pickle file using first command line argument
    dataset_artifact_file = sys.argv[1]
    # get model version from the parameters
    model_version = sys.argv[2]
    # loads the model parameters from the second command line argument
    model_params_file = sys.argv[3]

    # check if the files are valid
    if not dataset_artifact_file.endswith('.pkl') or not model_params_file.endswith('.json'):
        print('Please provide a valid pickle file and json file')
        return
    # check params json file exists
    if not os.path.exists(model_params_file):
        print(f'Model Params file: {model_params_file} not found')
        return

    # run training and evaluation
    model, dict_eval = train_and_evaluate(dataset_artifact_file, model_version, model_params_file)

    # pickle dict_eval to a file
    pickle.dump(dict_eval, open(f'evaluation_{model_version}.pkl', 'wb'))
    # pickle model into a file
    pickle.dump(model.model, open(f'model_{model_version}.pkl', 'wb'))

    print('Training completed successfully')
