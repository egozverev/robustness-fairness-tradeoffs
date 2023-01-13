from folktables import ACSDataSource, ACSIncome, BasicProblem, adult_filter
from sklearn.model_selection import train_test_split
import numpy as np

ACSIncomeSexGroup = BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
    ],
    target='PINCP',
    target_transform=lambda x: x > 50000,
    group='SEX',
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

if __name__ == "__main__":
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    ca_data = data_source.get_data(states=["CA"], download=True)
    features, label, group = ACSIncomeSexGroup.df_to_numpy(ca_data)
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
        features, label, group, test_size=0.85, random_state=42)  # taking only 15% of data
    group_train[group_train == 2] = 0
    np.save("folk_x.npy", X_train)
    np.save("folk_y.npy", y_train)
    np.save("folk_groups.npy", group_train)