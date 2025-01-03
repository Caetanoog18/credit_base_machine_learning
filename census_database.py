import pickle
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


class CensusData:

    def __init__(self):
        self.X_census_training = None
        self.X_census_test = None
        self.y_census_training = None
        self.y_census_test = None

    def census_database(self):

        with open('database/census.pkl', 'rb') as file:
            self.X_census_training,self.y_census_training, self.X_census_test, self.y_census_test = pickle.load(file)

        naive_bayes_census = GaussianNB()
        naive_bayes_census.fit(self.X_census_training, self.y_census_training)

        prediction = naive_bayes_census.predict(self.X_census_test)
        print(prediction)

        accuracy = accuracy_score(self.y_census_test, prediction)
        print(accuracy)

        confusion_matrix_ = confusion_matrix(self.y_census_test, prediction)
        print(confusion_matrix_)

        print(classification_report(self.y_census_test, prediction))

