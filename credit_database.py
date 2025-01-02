import pickle
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class CreditDatabase:

    def __init__(self):
        self.X_credit_training = None
        self.X_credit_test = None
        self.y_credit_training = None
        self.y_credit_test = None

    def credit_database(self):

        with open("database/credit_data.pkl", "rb") as file:
            self.X_credit_training, self.y_credit_training, self.X_credit_test, self.y_credit_test = pickle.load(file)

        #print(self.X_credit_training.shape, self.y_credit_training.shape)
        #print(self.X_credit_test.shape, self.y_credit_test.shape)

        naive_credit_database = GaussianNB()
        naive_credit_database.fit(self.X_credit_training, self.y_credit_training)

        prediction = naive_credit_database.predict(self.X_credit_test)
        #print(prediction)
        #print(self.y_credit_test)

        accuracy = accuracy_score(self.y_credit_test, prediction)
        print(accuracy)

        confusion_matrix_ = confusion_matrix(self.y_credit_test, prediction)
        print(confusion_matrix_)

        print(classification_report(self.y_credit_test, prediction))


