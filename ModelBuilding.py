import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib

# LinearRegression model training.
def LinerRegression_Model(X_train,Y_train,X_test,Y_test):
    pipeline_object=Pipeline([('Scaler',StandardScaler()),('Model_1',LinearRegression())])
    pipeline_object.fit(X_train,Y_train)
    print("\n .- LinearRegression Model Training Completed. ")
    # Basic Model Evaluvation for trainingSet.
    Y_train_Pred=pipeline_object.predict(X_train)
    rmse=np.sqrt(mean_squared_error(Y_train,Y_train_Pred))
    print('\n .- RMSE Training Error: ', int((rmse/Y_train.mean())*100),'%')
    Y_test_Pred=pipeline_object.predict(X_test)
    rmse=np.sqrt(mean_squared_error(Y_test,Y_test_Pred))
    print(' .- RMSE Testing Error: ', int((rmse/Y_test.mean())*100),'%')
    score=cross_val_score(pipeline_object,
                        X_train,
                        Y_train,
                        scoring="neg_mean_squared_error",
                        cv=5
                    )
    rmse_val=np.sqrt(-score)
    print('\n .- RMSE After validation: ', int((rmse_val.mean()/Y_train.mean())*100),'%')
    joblib.dump(pipeline_object,'Linear_regression_Model_v1.09.pkl')
    joblib.dump(X_train.columns.tolist(), 'liner_regressioncols.pkl')
    print('\n .- Model Deployed in Dev.')


# DecisionTreeRegressor Model training.
def DecisionTreeRegressor_model(X_train,Y_train,X_test,Y_test):
    pipeline_object2=Pipeline([('Scaler',StandardScaler()),('Model_1',DecisionTreeRegressor(
    max_depth=15, min_samples_split=23
))])
    pipeline_object2.fit(X_train,Y_train)
    print("\n .- DecisionTreeRegressor Model Training Completed. ")
    Y_train_Pred=pipeline_object2.predict(X_train)
    rmse=np.sqrt(mean_squared_error(Y_train,Y_train_Pred))
    print('\n .- RMSE Training Error: ', int((rmse/Y_train.mean())*100),'%')
    Y_test_Pred=pipeline_object2.predict(X_test)
    rmse=np.sqrt(mean_squared_error(Y_test,Y_test_Pred))
    print(' .- RMSE Testing Error: ', int((rmse/Y_test.mean())*100),'%')
    score=cross_val_score(pipeline_object2,
                        X_train,
                        Y_train,
                        scoring="neg_mean_squared_error",
                        cv=5
                    )
    rmse_val=np.sqrt(-score)
    print('\n .- RMSE After validation: ', int((rmse_val.mean()/Y_train.mean())*100),'%')
    joblib.dump(pipeline_object2,'Decision_Tree_Model_v1.09.pkl')
    joblib.dump(X_train.columns.tolist(), 'decision_treecols.pkl')
    print('\n .- Model Deployed in Dev.')



def RandomForestRegressor_model(X_train,Y_train,X_test,Y_test):
    pipeline_object3=Pipeline([('Scaler',StandardScaler()),('Model_3',RandomForestRegressor(
    n_estimators=200,
    max_depth=12,         
    min_samples_leaf=5,
    random_state=42
))])
    pipeline_object3.fit(X_train,Y_train)
    print("\n .- RandomForestRegressor Model Training Completed. ")
    Y_train_Pred=pipeline_object3.predict(X_train)
    rmse=np.sqrt(mean_squared_error(Y_train,Y_train_Pred))
    print('\n .- RMSE Training Error: ', int((rmse/Y_train.mean())*100),'%')
    Y_test_Pred=pipeline_object3.predict(X_test)
    rmse=np.sqrt(mean_squared_error(Y_test,Y_test_Pred))
    print(' .- RMSE Testing Error: ', int((rmse/Y_test.mean())*100),'%')
    score=cross_val_score(pipeline_object3,
                        X_train,
                        Y_train,
                        scoring="neg_mean_squared_error",
                        cv=5
                    )
    rmse_val=np.sqrt(-score)
    print('\n .- RMSE After validation: ', int((rmse_val.mean()/Y_train.mean())*100),'%')
    joblib.dump(pipeline_object3,'Random_forest_Model_v1.09.pkl')
    joblib.dump(X_train.columns.tolist(), 'random_forestcols.pkl')
    print('\n .- Model Deployed in Dev.')


