import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, KBinsDiscretizer
import pickle

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import StratifiedKFold

def import_data():
    train_path = "D:/Code/Projects/Bank Default Hackathon/Training Data.csv"
    test_path = "D:/Code/Projects/Bank Default Hackathon/Test Data.csv"

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Renaming the Id column to id in train set
    df_train.rename(columns={'Id':'id'},inplace=True)
    # tof = len(df_train)-28000
    # df_test = df_train.iloc[:28000,:]
    # df_train = df_train.iloc[28000:,:]

    print(df_train.shape, df_test.shape)
    return df_train, df_test

def preprocess(df_train, df_test):
    categorical_features = \
    ['married', 'house_ownership', 'car_ownership', 'profession', 'city', 'state']
    numerical_features = \
    ['id', 'income', 'age', 'experience', 'current_job_years', \
     'current_house_years', 'risk_flag']

    # Remove all text except characters
    for col in categorical_features:
        df_train[col] = df_train[col].apply(lambda x: re.sub(r'[^a-zA-Z]',' ',x).strip())
        df_test[col] = df_test[col].apply(lambda x: re.sub(r'[^a-zA-Z]',' ',x).strip())

    with open('profession.pickle', 'rb') as f:
        profession = pickle.load(f)
        f.close()
    with open('city.pickle', 'rb') as f:
        city = pickle.load(f)
        f.close()
    with open('state.pickle', 'rb') as f:
        state = pickle.load(f)
        f.close()

    ordinal_values = {
        'house_ownership' : {
            'owned': 2,
            'norent_noown': 0,
            'rented': 1
        },
        'profession':profession,
        'city': city,
        'state': state
    }
    nominal = ['married', 'car_ownership', 'profession', 'city', 'state']
    low_cardinality_nom = ['married', 'car_ownership']
    high_cardinality_nom = list(set(nominal)-set(low_cardinality_nom))
    low_cardinality_nom +=  ['house_ownership']
    
    # Label encoder for nominal high cardinality
    label_encoder = LabelEncoder()
    df_train_LE = df_train[high_cardinality_nom].copy()
    df_test_LE = df_test[high_cardinality_nom].copy()
    
    for col in high_cardinality_nom:
        df_train_LE[col] = label_encoder.fit_transform(df_train[col])
        # Get the dictionary to map the values
        le_dict = dict(zip(label_encoder.classes_, \
                           label_encoder.transform(label_encoder.classes_)))
        # We add -1 in case of unknown values
        df_test_LE[col] = df_test_LE[col].apply(lambda x: le_dict.get(x, -1))

    # One hot encoding for nominal features with low cardinality
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype='int')
    df_train_OHE = df_train[low_cardinality_nom].copy()
    df_test_OHE = df_test[low_cardinality_nom].copy()

    df_train_OHE = pd.DataFrame(OH_encoder.fit_transform(df_train[low_cardinality_nom]))
    df_test_OHE = pd.DataFrame(OH_encoder.transform(df_test[low_cardinality_nom]))

    df_train_OHE.index = df_train.index
    df_test_OHE.index = df_test.index
    df_train_OHE.columns = OH_encoder.get_feature_names(df_train[low_cardinality_nom].columns.tolist())
    df_test_OHE.columns = OH_encoder.get_feature_names(df_test[low_cardinality_nom].columns.tolist())

    # Encoding according to dictionary mapping for Ordinal features
    df_train_OF = pd.DataFrame()
    df_test_OF = pd.DataFrame()

    for col, mapping in ordinal_values.items():
        df_train_OF[col+'_OF'] = df_train[col].apply(lambda x: mapping.get(x, -1))
        df_test_OF[col+'_OF'] = df_test[col].apply(lambda x: mapping.get(x, -1))

    df_train_categorical = pd.concat([df_train_LE,df_train_OHE,df_train_OF],axis=1)
    df_test_categorical = pd.concat([df_test_LE,df_test_OHE,df_test_OF],axis=1)

    
    # Feature Generation
    df_train_numerical = df_train[numerical_features]
    numerical_features_test = numerical_features.copy()
    numerical_features_test.remove('risk_flag')
    df_test_numerical = df_test[numerical_features_test]
    numerical_features,df_test_numerical.columns.to_list()

    for feature1 in numerical_features:
        for feature2 in numerical_features:
            if feature1==feature2 or feature1=='risk_flag' or feature2=='risk_flag'\
            or feature1=='id' or feature2=='id':
    #             print(feature1,feature2)
                continue
            df_train_numerical.loc[:,feature1+'_by_'+feature2] = \
            df_train_numerical.loc[:,feature1]/df_train_numerical.loc[:,feature2]
            df_test_numerical.loc[:,feature1+'_by_'+feature2] = \
            df_test_numerical.loc[:,feature1]/df_test_numerical.loc[:,feature2]

            df_train_numerical.loc[:,feature1+'_into_'+feature2] = \
            df_train_numerical.loc[:,feature1]*df_train_numerical.loc[:,feature2]
            df_test_numerical.loc[:,feature1+'_into_'+feature2] = \
            df_test_numerical.loc[:,feature1]*df_test_numerical.loc[:,feature2]


    df_train_numerical.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_test_numerical.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_train_numerical.fillna(-1,inplace=True)
    df_test_numerical.fillna(-1,inplace=True)
    
    df_train_numerical['age_minus_experience'] = df_train_numerical['age']-\
    df_train_numerical['experience']
    df_train_numerical['experience_minus_current_job_years'] = df_train_numerical['experience']-\
    df_train_numerical['current_job_years']

    df_test_numerical['age_minus_experience'] = df_test_numerical['age']-\
    df_test_numerical['experience']
    df_test_numerical['experience_minus_current_job_years'] = df_test_numerical['experience']-\
    df_test_numerical['current_job_years']
    
    
    df_train_numerical['current_job_years_lte2'] = np.where(np.array(df_train_numerical['current_job_years'])<=2,1,0)
    df_test_numerical['current_job_years_lte2'] = np.where(np.array(df_test_numerical['current_job_years'])<=2,1,0)

    df_train_numerical['income_bnd'] = round(df_train['income']/400000)
    df_test_numerical['income_bnd'] = round(df_test['income']/400000)
    df_train_numerical['income_bnd'] = df_train_numerical['income_bnd'].astype(int)
    df_test_numerical['income_bnd'] = df_test_numerical['income_bnd'].astype(int)

    est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    cols_to_bin = ['age','experience', 'current_house_years', 'current_job_years']
    df_train_numerical[[i+'_bnd' for i in cols_to_bin]] = est.fit_transform(df_train[cols_to_bin])
    df_test_numerical[[i+'_bnd' for i in cols_to_bin]] = est.fit_transform(df_test[cols_to_bin])

    df_train_final = pd.concat([df_train_categorical, df_train_numerical],axis=1)
    df_test_final = pd.concat([df_test_categorical, df_test_numerical],axis=1)
    
    return df_train_final, df_test_final




# Function to get optimal model using grid search CV (n_splits = 5, validation set = 0.175 of train set)
def GrCV(param,clf,XY_data, sampler, scaler, search='grid'):
    [X_train, X_test, Y_train, Y_test] = XY_data
    X_train, Y_train = sampler.fit_resample(X_train, Y_train)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf.fit(X_train,Y_train)
    Y_train_pred = clf.predict(X_train)
    Y_test_pred = clf.predict(X_test)
    print("Default Training ROC AUC = {}".format(roc_auc_score(Y_train,Y_train_pred)))
    print("Default Test ROC AUC = {}".format(roc_auc_score(Y_test,Y_test_pred)))
    cv_split = StratifiedKFold(n_splits = 3, shuffle=True, random_state = 100 )
    if search == 'random':
        best_model = GridSearchCV(estimator = clf, param_distributions = param,\
                                    cv = cv_split, scoring = 'roc_auc')
    else:
        best_model = RandomizedSearchCV(estimator = clf, param_distributions = param,\
                                    cv = cv_split, scoring = 'roc_auc')
    best_model.fit(X_train, Y_train)
    best_param = best_model.best_params_
    print("Best parameters are : {}".format(best_param))
    clf.set_params(**best_param)
    clf.fit(X_train, Y_train)
    Y_train_pred = clf.predict(X_train)
    Y_test_pred = clf.predict(X_test)
    print("Training ROC AUC = {}".format(roc_auc_score(Y_train,Y_train_pred)))
    print("Test ROC AUC = {}".format(roc_auc_score(Y_test,Y_test_pred)))
    return clf, best_param

# Function to train the given model on given data and print the accuracies
def train(clf,XY_data):
    [X_train, X_test, Y_train, Y_test] = XY_data
    clf.fit(X_train,Y_train)

    return clf

# Function to train the given model on given data and print the accuracies
def predict(clf,XY_data,thresholds=[0.5]):
    [X_train, X_test, Y_train, Y_test] = XY_data
    pred_test = clf.predict_proba(X_test)[:,1]
    pred_train = clf.predict_proba(X_train)[:,1]
    for threshold in thresholds:
        pred_test_t = pred_test.copy()
        pred_train_t = pred_train.copy()
        pred_test_t[pred_test>=threshold] = 1
        pred_test_t[pred_test<threshold] = 0
        pred_train_t[pred_train>=threshold] = 1
        pred_train_t[pred_train_t<threshold] = 0

        test_roc = roc_auc_score(Y_test, pred_test_t)
        train_roc = roc_auc_score(Y_train, pred_train_t)
        print(f"Threshold = {threshold}")
        print("Default Training ROC AUC = {}".format(train_roc))
        print("Default Test ROC AUC = {}".format(test_roc))
#         print(classification_report(Y_test, pred_test_t))
        print(confusion_matrix(Y_test, pred_test_t))
        
        print()