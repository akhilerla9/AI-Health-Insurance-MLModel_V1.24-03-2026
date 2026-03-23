from sklearn.model_selection import StratifiedShuffleSplit

def train_test_split_data(df):

    split_object=StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=199)
    result=split_object.split(df,df['region'])
    for i,j in result:
        labelset_train= df.iloc[i]
        labelset_test= df.iloc[j]

    return labelset_train,labelset_test