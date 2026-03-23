import pandas as pd
import numpy as np
import joblib
from Preprocessing import data_transform,data_cleaning

TestCase=r"F:\Data Science + Gen AI\UseCases\Baxter_Health_Insurance_ML-Model\Bussiness Input\second_sample_test_cases.xlsx"


def processing(model,cols):
    SampleTestCase=data_cleaning(TestCase)
    X_Sample_Features,X_Sample_Labels,X_Sample_Features_copy,X_Sample_Labels_copy,=data_transform(SampleTestCase,SampleTestCase)
    # LOAD TRAINING COLUMNS
    cols = joblib.load(cols)

    # ADD MISSING COLUMNS = 0
    for c in cols:
        if c not in X_Sample_Features.columns:
            X_Sample_Features[c] = 0

    #  KEEP SAME ORDER
    X_Sample_Features = X_Sample_Features[cols]

    model_predict(X_Sample_Features, model)

def model_predict(X_Sample_Features,model):
    Y_Sample_Predict=model.predict(X_Sample_Features)
    X_Sample_Features['charges (Ai Generated)']=Y_Sample_Predict
    X_Sample_Features.to_excel(r'Bussiness Input\second_sample_test_cases_output.xlsx',sheet_name='AI_Generated_Prediction', index=False)
    print('\n .- AI Generated data Sucessfully Saved.')

def main():
    print('\n -> Baxter Health Insurance AI Model.v1.9 <-')
    userotp=int(input('\n .- LinearRegression [0] | DecissionTree [1] | RandomForest [2]: ',))

    if (userotp == 0):
        model_v1=joblib.load('Linear_regression_Model_v1.09.pkl')
        processing(model_v1,'liner_regressioncols.pkl')

    elif (userotp == 1):
        model_v2=joblib.load('Decision_Tree_Model_v1.09.pkl')
        processing(model_v2,'decision_treecols.pkl')
    elif (userotp == 2):
        model_v3=joblib.load('Random_forest_Model_v1.09.pkl')
        processing(model_v3,'random_forestcols.pkl')
    else:
        print('\n .- Invalid Model Selection.')
if __name__=='__main__':
    main()