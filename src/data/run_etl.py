# Runs the ETL process
#    gets the filename from arguments in command line
# ------------------------------------------------------------------------
# Author: Moacir A. Ponti
# 2024
# ------------------------------------------------------------------------

import sys
from etl import ETL


def main():
    # get the csv filename from the command line arguments
    csv_file = sys.argv[1]
    # get the name of the dataset from the command line arguments
    dataset_name = sys.argv[2]
    # get the version of the dataset from the command line arguments
    dataset_version = sys.argv[3]

    # check the name and version are provided
    if not dataset_name or not dataset_version or not csv_file:
        print('Please provide the csv file, dataset name and version of the dataset as follows:')
        print('python run_etl.py <csv_file> <dataset_name> <dataset_version>')
        return

    # check if the csv file is valid
    if not csv_file.endswith('.csv'):
        print(f'Please provide a valid csv file, {csv_file} is not a valid csv file')
        return

    # create an instance of the ETL class
    etl = ETL()
    # load the dataset from the csv file
    dataset = etl.load_dataset_from_csv('data', csv_file)
    etl.select_columns()

    # validate dataset
    if not etl.validate_dataset(dataset):
        print('ETL process failed')
        return

    # dictionary containing dataset information
    dataset_artifact = {'name': dataset_name,
                        'version': dataset_version,
                        'dataset': dataset,
                        'features': etl.selected_features}

    etl.serialize_dataset(dataset_artifact)

    print('ETL process completed successfully')
