import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class PreprocessingCredit:

    def __init__(self):
        self.credit_data = None
        self.X_credit = None
        self.y_credit = None
        self.X_credit_training = None
        self.y_credit_training = None
        self.X_credit_test = None
        self.y_credit_test = None



    def pre_processing(self):

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