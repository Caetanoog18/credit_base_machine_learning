import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

class RiskCreditBase:

    def __init__(self):
        self.credit_risk_database = pd.read_csv('database/credit_risk.csv')
        self.X_credit_risk = None
        self.y_credit_risk = None

    def naive_bayes(self):
        self.X_credit_risk = self.credit_risk_database.iloc[:, 0:4].values
        self.y_credit_risk = self.credit_risk_database.iloc[:, 4].values

        label_encoder_history = LabelEncoder()
        label_encoder_debt = LabelEncoder()
        label_encoder_guarantee = LabelEncoder()
        label_encoder_income = LabelEncoder()

        self.X_credit_risk[:, 0] = label_encoder_history.fit_transform(self.X_credit_risk[:, 0])
        self.X_credit_risk[:, 1] = label_encoder_debt.fit_transform(self.X_credit_risk[:, 1])
        self.X_credit_risk[:, 2] = label_encoder_guarantee.fit_transform(self.X_credit_risk[:, 2])
        self.X_credit_risk[:, 3] = label_encoder_income.fit_transform(self.X_credit_risk[:, 3])

        #print(self.X_credit_risk)

        with open('database/credit_risk.pkl', 'wb') as file:
            pickle.dump([self.X_credit_risk, self.y_credit_risk], file)

        naive_credit_risk = GaussianNB()
        naive_credit_risk.fit(self.X_credit_risk, self.y_credit_risk)

        prediction = naive_credit_risk.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
        # print(prediction)
        # print(naive_credit_risk.classes_)
        # print(naive_credit_risk.class_count_)
        # print(naive_credit_risk.class_prior_)







