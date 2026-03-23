import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

def data_cleaning(dataset):

    # loading the dataset.
    df=pd.read_excel(dataset)

    # Hadiling gener, smoker, chronic_disease columns from string to int.
    df['gender']=df['gender'].replace({'male':1,'female':0})
    df['smoker']=df['smoker'].replace({'yes':1,'no':0})
    df['chronic_disease']=df['chronic_disease'].replace({'yes':1,'no':0})

    return df


def data_transform(labelset_train, labelset_test):

    # trainset_Implementation for converting String data to Numerical data.
    # label Saperation.
    Y_train=labelset_train['charges']
    X_train_numerical_features= labelset_train[
        ['age','gender','bmi','children','smoker','income','chronic_disease','hospital_visits_last_year',]
        ]
    X_train_string_feature=labelset_train[
        ['region','occupation','exercise_frequency','policy_type']
        ]
    # We are using OneHotEncoder to conver the categorical data in to integer.
    encoder_object=OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    sample=encoder_object.fit_transform(X_train_string_feature)
    sampleTemp=pd.DataFrame(sample,
                        columns=encoder_object.get_feature_names_out(
                            ['region','occupation','exercise_frequency','policy_type']
                        ),
                        index=X_train_string_feature.index)

    X_train=pd.concat([X_train_numerical_features,sampleTemp],axis=1)

    # similarly testingset_Implementation for converting String data to Numerical data.

    # label Saperation.
    Y_test=labelset_test['charges']
    X_test_numerical_features= labelset_test[
        ['age','gender','bmi','children','smoker','income','chronic_disease','hospital_visits_last_year',]
        ]
    X_test_string_feature=labelset_test[
        ['region','occupation','exercise_frequency','policy_type']
        ]
    # We are using OneHotEncoder to conver the categorical data in to integer.
    sample=encoder_object.transform(X_test_string_feature)
    sampleTemp=pd.DataFrame(sample,
                        columns=encoder_object.get_feature_names_out(
                            ['region','occupation','exercise_frequency','policy_type']
                        ),
                        index=X_test_string_feature.index)

    X_test=pd.concat([X_test_numerical_features,sampleTemp],axis=1)

    return X_train,Y_train, X_test,Y_test 

