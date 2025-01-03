import tensorflow as tf
from risk_credit_database import RiskCreditBase
from pre_processing import PreprocessingCredit, Census
from credit_database import CreditDatabase
from census_database import CensusData

risk_credit = RiskCreditBase()
risk_credit.naive_bayes()

credit_pre_processing = PreprocessingCredit()
credit_pre_processing.pre_processing_credit()

credit = CreditDatabase()
credit.credit_database()

census = Census()
census.pre_processing_census()

census_data = CensusData()
census_data.census_database()





