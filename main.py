import tensorflow as tf
from risk_credit_database import RiskCreditBase
from pre_processing import PreprocessingCredit
from credit_database import CreditDatabase

risk_credit = RiskCreditBase()
risk_credit.naive_bayes()

credit_pre_processing = PreprocessingCredit()
credit_pre_processing.pre_processing()

credit = CreditDatabase()
credit.credit_database()





