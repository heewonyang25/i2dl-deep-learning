import numpy as np
import os
from exercise_code.data.csv_dataset import CSVDataset
from exercise_code.data.csv_dataset import FeatureSelectorAndNormalizationTransform
from exercise_code.data.dataloader import DataLoader


def binarize(X, y, a_percentile, b_percentile):
    """ Splits data to be smaller than the a_percentil and larger than b_percentile
    :param x: input
    :param y: labels
    :param a_percentile:
    :param b_percentile:
    :return:
    :rtype: X, Y
    """
    data_index = ((a_percentile >= y) | (y >= b_percentile))
    y = y[data_index]
    x = X[data_index[:, 0]]

    y[y <= a_percentile] = 0
    y[y >= b_percentile] = 1

    return x, np.expand_dims(y, 1)


def test_accuracy(y_pred, y_true):
    """ Compute test error / accuracy
    Params:
    ------
    y_pred: model prediction
    y_true: ground truth values
    return:
    ------
    Accuracy / error on test set
    """

    # Apply threshold
    threshold = 0.50

    y_binary = np.zeros_like((y_pred))
    y_binary[y_pred >= threshold] = 1
    y_binary[y_pred < threshold] = 0

    # Get final predictions.
    y_binary = y_binary.flatten().astype(int)
    y_true = y_true.flatten().astype(int)

    acc = (y_binary == y_true).mean()
    return acc

def get_housing_data(train_dataset=None, target_column=None):
    # Load the data into datasets/housing
    i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
    root_path = os.path.join(i2dl_exercises_path, "datasets", 'housing')
    housing_file_path = os.path.join(root_path, "housing_train.csv")
    download_url = 'https://i2dl.dvl.in.tum.de/downloads/housing_train.zip'

    # Always make sure this line was run at least once before trying to
    # access the data manually, as the data is downloaded in the
    # constructor of CSVDataset.
    target_column = 'SalePrice'
    train_dataset = CSVDataset(target_column=target_column, root=root_path, download_url=download_url, mode="train")

    #Compute the min, max and mean
    df = train_dataset.df
    # Select a feature to keep plus the target column.
    selected_columns = ['GrLivArea', target_column]
    mn, mx, mean = df.min(), df.max(), df.mean()

    column_stats = {}
    for column in selected_columns:
        crt_col_stats = {'min': mn[column],
                         'max': mx[column],
                         'mean': mean[column]}
        column_stats[column] = crt_col_stats

    transform = FeatureSelectorAndNormalizationTransform(column_stats, target_column)

    #loading our train, val and test datasets
    train_dataset = CSVDataset(mode="train", target_column=target_column, root=root_path, download_url=download_url,
                               transform=transform)
    val_dataset = CSVDataset(mode="val", target_column=target_column, root=root_path, download_url=download_url,
                             transform=transform)
    test_dataset = CSVDataset(mode="test", target_column=target_column, root=root_path, download_url=download_url,
                              transform=transform)

    #Store our data in a matrix
    # load training data into a matrix of shape (N, D), same for targets resulting in the shape (N, 1)
    X_train = [train_dataset[i]['features'] for i in range((len(train_dataset)))]
    X_train = np.stack(X_train, axis=0)
    y_train = [train_dataset[i]['target'] for i in range((len(train_dataset)))]
    y_train = np.stack(y_train, axis=0)

    # load validation data
    X_val = [val_dataset[i]['features'] for i in range((len(val_dataset)))]
    X_val = np.stack(X_val, axis=0)
    y_val = [val_dataset[i]['target'] for i in range((len(val_dataset)))]
    y_val = np.stack(y_val, axis=0)

    # load train data
    X_test = [test_dataset[i]['features'] for i in range((len(test_dataset)))]
    X_test = np.stack(X_test, axis=0)
    y_test = [test_dataset[i]['target'] for i in range((len(test_dataset)))]
    y_test = np.stack(y_test, axis=0)

    #label the data for our classification problem
    y_all = np.concatenate([y_train, y_val, y_test])
    thirty_percentile = np.percentile(y_all, 30)
    seventy_percentile = np.percentile(y_all, 70)

    # Prepare the labels for classification.
    X_train, y_train = binarize(X_train, y_train, thirty_percentile, seventy_percentile)
    X_val, y_val = binarize(X_val, y_val, thirty_percentile, seventy_percentile)
    X_test, y_test = binarize(X_test, y_test, thirty_percentile, seventy_percentile)

    print('You successfully loaded your data! \n')

    return X_train, y_train, X_val, y_val, X_test, y_test, train_dataset



