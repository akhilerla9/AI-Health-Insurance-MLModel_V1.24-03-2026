'''
Problem Satement: Aim is to predict the health insurance charges (Yearly plan) based on user life style.
'''
import pandas as pd
from Preprocessing import data_cleaning,data_transform
from TrainTestSplitData import train_test_split_data
from ModelBuilding import LinerRegression_Model,DecisionTreeRegressor_model,RandomForestRegressor_model

dataset_Env=r"F:\Data Science + Gen AI\UseCases\Baxter_Health_Insurance_ML-Model\Insurance_RealBehaviour.xlsx"
class HealthInsurance_Solution:
    
    # Data Processing.
    def data_preprocessing(self):
        self.dataset=data_cleaning(self.dataset)

    # Spliting dataset.
    def split_train_test(self):
        self.labelset_train,self.labelset_test=train_test_split_data(self.dataset)

    # data Transformation 
    def data_transform(self):
        self.X_train,self.Y_train, self.X_test,self.Y_test = data_transform(self.labelset_train,self.labelset_test)

    # Model Finalization & Accurace details.
    def Model_building(self): 
        RandomForestRegressor_model(self.X_train,self.Y_train,self.X_test,self.Y_test)



    # Ready for user Queries.

    def __init__(self,dataset):
        self.dataset=dataset
    pass

def main():
    print('\n -> Baxter AI Health Insurance <- \n')
    soluition_instnace=HealthInsurance_Solution(dataset_Env)
    soluition_instnace.data_preprocessing()
    soluition_instnace.split_train_test()
    soluition_instnace.data_transform()
    soluition_instnace.Model_building()
if __name__=='__main__':
    main()