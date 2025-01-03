import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


class PreprocessingCredit:

    def __init__(self):
        self.credit_data = None
        self.X_credit = None
        self.y_credit = None
        self.X_credit_training = None
        self.y_credit_training = None
        self.X_credit_test = None
        self.y_credit_test = None



    def pre_processing_credit(self):

        self.credit_data = pd.read_csv('database/credit_data.csv')

        # Verifying inconsistent values
        print(self.credit_data.loc[self.credit_data['age']<0])

        # Mean without inconsistent values
        mean = self.credit_data["age"][self.credit_data["age"]>0].mean()

        self.credit_data.loc[self.credit_data['age']<0, 'age'] = mean

        # Missing values
        print(self.credit_data.loc[pd.isnull(self.credit_data['age'])])
        self.credit_data.fillna(self.credit_data["age"].mean(), inplace=True)

        self.X_credit = self.credit_data.iloc[:, 1:4].values
        self.y_credit = self.credit_data.iloc[:, 4].values

        # Printing the maximum and the minimum values
        print(self.X_credit[:, 0].min(), self.X_credit[:, 1].min(), self.X_credit[:, 2].min())
        print(self.X_credit[:, 0].max(), self.X_credit[:, 1].max(), self.X_credit[:, 2].max())

        credit_scaler = StandardScaler()
        self.X_credit = credit_scaler.fit_transform(self.X_credit)

        # Printing the maximum and the minimum values
        print(self.X_credit[:, 0].min(), self.X_credit[:, 1].min(), self.X_credit[:, 2].min())
        print(self.X_credit[:, 0].max(), self.X_credit[:, 1].max(), self.X_credit[:, 2].max())

        self.X_credit_training, self.y_credit_training, self.X_credit_test, self.y_credit_test = train_test_split(self.X_credit, self.y_credit, random_state = 42, test_size = 0.25)

        with open('database/credit_data.pkl', 'wb') as file:
            pickle.dump([self.X_credit_training, self.X_credit_test, self.y_credit_training, self.y_credit_test], file)

class Census:
    def __init__(self):
        self.census_data = None
        self.X_census = None
        self.y_census = None
        self.X_census_training = None
        self.X_census_test = None
        self.y_census_training = None
        self.y_census_test = None

    def pre_processing_census(self):

        self.census_data = pd.read_csv('database/census.csv')

        self.X_census = self.census_data.iloc[:, 0:14].values
        self.y_census = self.census_data.iloc[:, 14].values

        label_encoder_workclass = LabelEncoder()
        label_encoder_education = LabelEncoder()
        label_encoder_marital = LabelEncoder()
        label_encoder_occupation = LabelEncoder()
        label_encoder_relationship = LabelEncoder()
        label_encoder_race = LabelEncoder()
        label_encoder_sex = LabelEncoder()
        label_encoder_country = LabelEncoder()

        # Accessing x_census and applying the label encoder
        self.X_census[:, 1] = label_encoder_workclass.fit_transform(self.X_census[:,1])
        self.X_census[:, 3] = label_encoder_education.fit_transform(self.X_census[:,3])
        self.X_census[:, 5] = label_encoder_marital.fit_transform(self.X_census[:,5])
        self.X_census[:, 6] = label_encoder_occupation.fit_transform(self.X_census[:,6])
        self.X_census[:, 7] = label_encoder_relationship.fit_transform(self.X_census[:,7])
        self.X_census[:, 8] = label_encoder_race.fit_transform(self.X_census[:,8])
        self.X_census[:, 9] = label_encoder_sex.fit_transform(self.X_census[:,9])
        self.X_census[:, 13] = label_encoder_country.fit_transform(self.X_census[:,13])



        one_hot_encoder_census = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder="passthrough")

        self.X_census = one_hot_encoder_census.fit_transform(self.X_census).toarray()


        scaler_census = StandardScaler()

        self.X_census = scaler_census.fit_transform(self.X_census)

        self.X_census_training, self.X_census_test, self.y_census_training, self.y_census_test = train_test_split(self.X_census, self.y_census, test_size=0.15, random_state=0)

        print(self.X_census_training.shape, self.y_census_training.shape)
        print(self.X_census_test.shape, self.y_census_test.shape)

        with open('database/census.pkl', 'wb') as file:
            pickle.dump([self.X_census_training, self.y_census_training, self.X_census_test, self.y_census_test], file)
