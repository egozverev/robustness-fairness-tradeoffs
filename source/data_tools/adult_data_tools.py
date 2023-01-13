import numpy as np
import pandas as pd

from sklearn import preprocessing

# Note: one of the reasons we are using folktables is that it's a much cleaner dataset.
def get_preprocessed_adult_data(custom_file_address=False, train_file=None, test_file=None):
    """
    Preprocessing for adult dataset.
    Inspired by some kaggle competition notebooks.
    :param custom_file_address: if False, download dataset from internet, else from file
    :param train_file: (optional) train data
    :param test_file: (optional) test data
    :return:
    """
    features = ["age", "workclass", "fnlwgt", "education", "educational-num", "marital-status",
                "occupation", "relationship", "race", "gender", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"]
    if not custom_file_address:
        train_file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
        test_file = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'

    original_train = pd.read_csv(train_file, names=features, sep=r'\s*,\s*',
                                 engine='python', na_values="?")
    # This will download 1.9M
    original_test = pd.read_csv(test_file, names=features, sep=r'\s*,\s*',
                                engine='python', na_values="?", skiprows=1)

    # data = pd.concat([original_train, original_test])
    datasets = []
    for data in [original_train, original_test]:
        data['income'] = data['income'].replace('<=50K', 0).replace('>50K', 1)
        data['income'] = data['income'].replace('<=50K.', 0).replace('>50K.', 1)
        del data['education']
        data = data.drop("fnlwgt", axis=1)
        data = data.drop_duplicates()
        data['workclass'] = data['workclass'].replace("?", "None")
        label_encoder = preprocessing.LabelEncoder()
        data['workclass'] = label_encoder.fit_transform(data['workclass'])
        data['marital-status'] = data['marital-status'].map({"Never-married": 1, "Separated": 2, "Widowed": 3,
                                                             "Married-spouse-absent": 4, "Married-AF-spouse": 5,
                                                             "Divorced": 6, "Married-civ-spouse": 7})
        data['occupation'] = data['occupation'].replace("?", "None")
        data['occupation'] = label_encoder.fit_transform(data['occupation'])
        data['relationship'] = label_encoder.fit_transform(data['relationship'])
        data['race'] = label_encoder.fit_transform(data['race'])
        data['gender'] = label_encoder.fit_transform(data['gender'])
        data['native-country'] = data['native-country'].replace("?", "None")
        data['native-country'] = label_encoder.fit_transform(data['native-country'])
        x = data[
            ['age', 'workclass', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender',
             'capital-gain', 'capital-loss', 'native-country']].values  # independent features
        y = data['income'].values  # y -> target/true labels
        groups = x[:, 7]
        datasets += [x, y, groups]
    return datasets  # X_train, y_train, groups_train, X_test, y_test, groups_test
