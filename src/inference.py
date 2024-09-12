# In inference, loads model from pickle, and that predicts with a single data
# ------------------------------------------------------------------------
# Author: Moacir A. Ponti
# 2024
# ------------------------------------------------------------------------


import sys
import json

from model.model import Model


def load_model(model_file):
    """
    Loads the model from a pickle file
    Args:
        model_file: str - the path to the pickle file
    Returns:
        model: the model
    """
    model = Model.load(model_file)
    return model


def predict(model, data):
    """
    Predicts the target
    Args:
        model: the model
        data: np.array - the data to predict
    Returns:
        np.array: the predictions
    """

    # check if json contain all the features as defined in model.features, and if not imput with NULL
    # predict
    scores, missing_features = model.predict(data)

    response = {'score': round(float(scores[1]), 4),
                'no_missing_features': missing_features}

    return response


def main():
    # loads the model from the pickle file using first command line argument
    model_file = sys.argv[1]
    # loads the data from a JSON file from the second command line argument
    data_file = sys.argv[2]
    # check if the files are valid
    if not model_file.endswith('.pkl') and not data_file.endswith('.json'):
        print('Please provide a valid pickle file and a data as a json file')
        return

    # loads model from pickle
    model = load_model(model_file)
    # loads data from json
    with open(data_file, 'r') as file:
        data = json.load(file)

    print(f'Inference with model {model.name}, version {model.version}...')
    # predict
    response = predict(model, data)

    print(response)