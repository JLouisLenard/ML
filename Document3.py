import tensorflow as tf
import sys
import time
import traceback

from pandas import DataFrame
import pyodbc  
from scipy import sparse
from scipy.sparse.coo import coo_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model.base import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.regression import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import StandardScaler, OneHotEncoder
from sklearn.preprocessing.label import LabelEncoder
from sklearn.svm import SVR
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.tree.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DF_column_names = ['IDentity','CreateDTS','UpdateDTS','CPT','Mod1', 'Mod2','Mod3', 'Mod4', 'ICD1', 'ICD2', 'ICD3', 'ICD4', 
                'ChargeAmount', 'ClaimStatus', 'PayerCategory', 'PayerState', 'Specialty', 'Sex', 'FacilityType', 'InsuranceType',
                'DOSFrom', 'DOSTo','SubmitDate', 'PaidAmount', 'R1', 'R2', 'R3', 'R4', 'R5','R6','R7','R8','R9', 'R10',
                'AdjCode1', 'AdjAmt1', 'AdjCode2', 'AdjAmt2', 'AdjCode3', 'AdjAmt3', 'AdjCode4', 'AdjAmt4', 'AdjCode5', 'AdjAmt5',
                'AdjCode6', 'AdjAmt6', 'AdjCode7', 'AdjAmt7', 'AdjCode8', 'AdjAmt8', 'AdjCode9', 'AdjAmt9', 'AdjCode10', 'AdjAmt10']

DF_cat_features = ['CPT','Mod1', 'Mod2','Mod3', 'Mod4', 'ICD1', 'ICD2', 'ICD3', 'ICD4', 
                   'PayerCategory','PayerState','Specialty', 'Sex', 'InsuranceType', 'IDentity']
                
                
DF_cat_labels = ['ClaimStatus','R1', 'R2', 'R3', 'R4',
                'AdjCode1', 'AdjCode2', 'AdjCode3', 'AdjCode4', 'AdjCode5']

DF_numeric_features = ['ChargeAmount']
DF_numeric_labels   = ['PaidAmount','AdjAmt1', 'AdjAmt2', 'AdjAmt3', 'AdjAmt4', 'AdjAmt5']
MIN_DATA_THRESHOLD = 100
TEST_FRACTION = 0.1
cleanup_columns =['PayerState', 'CPT']
NUMERIC = 'Numeric'
FACILITY_TYPE = 'FacilityType'
PAYER_STATE = 'PayerState'
PAYER_CAT = 'PayerCategory'
sql_column_names0 = 'ClaimDataID,CreatorDTS,UpdateDTS,CPTCode,Modifier1, Modifier2,Modifier3, Modifier4, DiagnosisCode1, DiagnosisCode2,DiagnosisCode3,DiagnosisCode4,'   
sql_column_names1 = 'ChargeAmount, ClaimStatus, PayerCatagory,PayerState,Specialty,PatientGender, FacilityType, InsuranceType,DOSFrom, DosTo,' 
sql_column_names2 = 'ClaimSubmitDate,InsurancePaidAmount, ClaimRejectionCode1, ClaimRejectionCode2, ClaimRejectionCode3, ClaimRejectionCode4, ClaimRejectionCode5,'   
sql_column_names3 = 'ClaimRejectionCode6, ClaimRejectionCode7, ClaimRejectionCode8, ClaimRejectionCode9, ClaimRejectionCode10,'   
sql_column_names4 = 'ClaimAdjustmentCode1, ClaimAdjustmentAmount1, ClaimAdjustmentCode2, ClaimAdjustmentAmount2, ClaimAdjustmentCode3, ClaimAdjustmentAmount3, ClaimAdjustmentCode4, ClaimAdjustmentAmount4, ClaimAdjustmentCode5, ClaimAdjustmentAmount5,' 
sql_column_names5 = 'ClaimAdjustmentCode6, ClaimAdjustmentAmount6, ClaimAdjustmentCode7, ClaimAdjustmentAmount7, ClaimAdjustmentCode8, ClaimAdjustmentAmount8, ClaimAdjustmentCode9, ClaimAdjustmentAmount9, ClaimAdjustmentCode10, ClaimAdjustmentAmount10' 


class DataFrameSelectorAsCategory(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self;
    def transform(self, X):
        return X[self.attribute_names].astype('category').values

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self;
    def transform(self, X):
        return X[self.attribute_names].astype('|S').values
         
class DataFrameNumericCleanerSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self;
    def transform(self, X):
        return np.nan_to_num(X[self.attribute_names].values)


def get_one_hot_category_data(selector, df):
    hot_encoder = OneHotEncoder()
    category_encoded= get_label_encoded_category_data(selector, df)
    category_hot_encoded = hot_encoder.fit_transform(category_encoded.reshape(-1,1))
    return category_hot_encoded

def get_label_encoded_category_data(selector, df):
    encoder = LabelEncoder()
    category_clean = selector.fit_transform(df)
    category_encoded= encoder.fit_transform(category_clean)
    return category_encoded

def get_data_frame(category, state, df):
    category_df = df[df[PAYER_CAT == category]]
    state_df = category_df[category_df == state] 
    return state_df, category_df

def clean_data_frame(df):
# remove data that is below threshold count
    for column in cleanup_columns:
        col_value_counts=df[column].value_counts()
        col_values_to_remove = col_value_counts[col_value_counts < MIN_DATA_THRESHOLD].index
        for value in col_values_to_remove:
            valueIndex = df[df[column]== value].index
            df = df.drop(valueIndex)

    return df

def prepare_one_hot_encoded_data(df):

    #now create a dataframe for each state before proceeding

    #cleanup and scale the numerical data 
    num_feature_pipeline = Pipeline([
        ('selector', DataFrameNumericCleanerSelector(DF_numeric_features)),
        ('std_scaler', StandardScaler())
        ])
    num_label_pipeline = Pipeline([
        ('selector', DataFrameNumericCleanerSelector(DF_numeric_labels)),
        ])
    numeric_features = {}
    claims_labels_dict = {}
    claims_numeric_labels = {}
    claims_feature_encoded_coo = {}
    claims_data_encoded_coo = {}

    for payer_cat in payer_cat_value_count.index:
        cat_df = df[df[PAYER_CAT] == payer_cat]
        numeric_features[payer_cat] = {}
        claims_labels_dict[payer_cat] = {}
        claims_numeric_labels[payer_cat] = {}
        claims_feature_encoded_coo[payer_cat] = {}
        claims_data_encoded_coo[payer_cat] = {}
        for state in state_value_count.index:
            if state != 0:
                print(state)

                state_df = cat_df[cat_df[PAYER_STATE] == state]  
                if state_df.shape[0] > 1000 :
                    numeric_features[payer_cat][state] = num_feature_pipeline.fit_transform(state_df)
    
                    claims_feature_encoded_coo[payer_cat][state] = coo_matrix(arg1=numeric_features[payer_cat][state])
                    claims_numeric_labels[payer_cat][state] = num_label_pipeline.fit_transform(state_df)
    
                    #create a dictionary so that the data can be grouped based on various labels
                    claims_labels_dict[payer_cat][state] = {}
                    claims_labels_dict[payer_cat][state][NUMERIC] = claims_numeric_labels[payer_cat][state]
                    #one hot encode the categorial labels and features
                    for category in DF_cat_labels:
                        print(category)
                        selector = DataFrameSelector(category)
                        claims_labels_dict[payer_cat][state][category] = get_one_hot_category_data(selector, state_df)
                    for category in DF_cat_features:
                        print(category)
                        selector = DataFrameSelector(category)
                        claims_feature_encoded_coo[payer_cat][state] = sparse.hstack([claims_feature_encoded_coo[payer_cat][state],get_one_hot_category_data(selector,state_df)])

                    selector = DataFrameSelectorAsCategory(FACILITY_TYPE)
                    claims_data_encoded_coo[payer_cat][state] = sparse.hstack([claims_feature_encoded_coo[payer_cat][state],get_one_hot_category_data(selector, state_df)])

    return claims_data_encoded_coo, claims_labels_dict
    
def prepare_one_hot_encoded_data_full(df):

    #now create a dataframe for each state before proceeding

    #cleanup and scale the numerical data 
    num_feature_pipeline = Pipeline([
        ('selector', DataFrameNumericCleanerSelector(DF_numeric_features)),
        ('std_scaler', StandardScaler())
        ])
    num_label_pipeline = Pipeline([
        ('selector', DataFrameNumericCleanerSelector(DF_numeric_labels)),
        ])
    claims_labels_dict = {}
    numeric_features = num_feature_pipeline.fit_transform(df)
    claims_feature_encoded_coo = coo_matrix(arg1=numeric_features)
    #create a dictionary so that the data can be grouped based on various labels
    claims_labels_dict[NUMERIC] = num_label_pipeline.fit_transform(df)
    
    #one hot encode the categorial labels and features
    for category in DF_cat_labels:
        print(category)
        selector = DataFrameSelector(category)
        claims_labels_dict[category] = get_one_hot_category_data(selector, df)
        print('label shape: ', claims_label_dict[category].shape)
    for category in DF_cat_features:
        print(category)
        selector = DataFrameSelector(category)
        temp = get_one_hot_category_data(selector,df);
        print('feature shape: ', temp.shape)
        claims_feature_encoded_coo = sparse.hstack([claims_feature_encoded_coo,temp])

    selector = DataFrameSelectorAsCategory(FACILITY_TYPE)
    claims_data_encoded_coo = sparse.hstack([claims_feature_encoded_coo,get_one_hot_category_data(selector, df)])

    return claims_data_encoded_coo, claims_labels_dict
    
def prepare_data(df):

    #now create a dataframe for each state before proceeding

    #cleanup and scale the numerical data 
    num_feature_pipeline = Pipeline([
        ('selector', DataFrameNumericCleanerSelector(DF_numeric_features)),
        ])
    num_label_pipeline = Pipeline([
        ('selector', DataFrameNumericCleanerSelector(DF_numeric_labels)),
        ])
    claims_labels_dict = {}
    claims_numeric_labels = {}
    claims_feature_encoded = {}
    claims_data_encoded = {}

    for payer_cat in payer_cat_value_count.index:
        cat_df = df[df[PAYER_CAT] == payer_cat]
        claims_labels_dict[payer_cat] = {}
        claims_numeric_labels[payer_cat] = {}
        claims_feature_encoded[payer_cat] = {}
        claims_data_encoded[payer_cat] = {}
        for state in state_value_count.index:
            if state != 0:
                print(state)

                state_df = cat_df[cat_df[PAYER_STATE] == state]  
                if state_df.shape[0] > 1000 :
                    
    
                    claims_feature_encoded[payer_cat][state] = num_feature_pipeline.fit_transform(state_df)
                    claims_numeric_labels[payer_cat][state] = num_label_pipeline.fit_transform(state_df)
    
                    #create a dictionary so that the data can be grouped based on various labels
                    claims_labels_dict[payer_cat][state] = {}
                    claims_labels_dict[payer_cat][state][NUMERIC] = claims_numeric_labels[payer_cat][state]
                    #one hot encode the categorial labels and features
                    for category in DF_cat_labels:
                        print(category)
                        selector = DataFrameSelector(category)
                        claims_labels_dict[payer_cat][state][category] = get_label_encoded_category_data(selector, state_df)
                    for category in DF_cat_features:
                        print(category)
                        selector = DataFrameSelector(category)
                        temp = get_label_encoded_category_data(selector,state_df).reshape(-1,1)
                        claims_feature_encoded[payer_cat][state] = np.hstack([claims_feature_encoded[payer_cat][state],temp])

                    selector = DataFrameSelectorAsCategory(FACILITY_TYPE)
                    temp = np.asarray(get_label_encoded_category_data(selector,state_df)).reshape(-1,1)
                    claims_data_encoded[payer_cat][state] = np.hstack([claims_feature_encoded[payer_cat][state],temp])
    return claims_data_encoded, claims_labels_dict

def prepare_data_full(df):

    #now create a dataframe for each state before proceeding

    #cleanup and scale the numerical data 
    num_feature_pipeline = Pipeline([
        ('selector', DataFrameNumericCleanerSelector(DF_numeric_features)),
        ])
    num_label_pipeline = Pipeline([
        ('selector', DataFrameNumericCleanerSelector(DF_numeric_labels)),
        ])
    claims_labels_dict = {}
    claims_feature_encoded = num_feature_pipeline.fit_transform(df)
    claims_numeric_labels = num_label_pipeline.fit_transform(df)
    #create a dictionary so that the data can be grouped based on various labels
    claims_labels_dict[NUMERIC] = claims_numeric_labels
    #one hot encode the categorial labels and features
    for category in DF_cat_labels:
        print(category)
        selector = DataFrameSelector(category)
        claims_labels_dict[category] = get_label_encoded_category_data(selector, df)
        print('label shape: ', claims_label_dict[category].shape)
    for category in DF_cat_features:
        print(category)
        selector = DataFrameSelector(category)
        temp = get_label_encoded_category_data(selector,df).reshape(-1,1)
        print('feature shape: ', temp.shape)
        claims_feature_encoded = np.hstack([claims_feature_encoded,temp])

    selector = DataFrameSelectorAsCategory(FACILITY_TYPE)
    temp = np.asarray(get_label_encoded_category_data(selector,df)).reshape(-1,1)
    claims_data_encoded = np.hstack([claims_feature_encoded,temp])
    return claims_data_encoded, claims_labels_dict

def prepare_train_label_data(claims_data_encoded, claims_labels_dict):
    train_set = {}
    test_set = {}
    y_train = {}
    y_test = {}
    dict_labels = DF_cat_labels
    dict_labels.append(NUMERIC)
    for payer_cat in payer_cat_value_count.index:
        train_set[payer_cat] = {}
        test_set[payer_cat] = {}
        y_train[payer_cat] = {}
        y_test[payer_cat] = {}
        for state in state_value_count.index:
            # #############################################################################
            # create test features and labels
            if state != 0:
                train_set[payer_cat][state] = {}
                test_set[payer_cat][state] = {}
                y_train[payer_cat][state] = {}
                y_test[payer_cat][state] = {}
                for label in dict_labels: 
                    print(label)
                    try:
                        train_set[payer_cat][state][label], test_set[payer_cat][state][label],y_train[payer_cat][state][label], y_test[payer_cat][state][label] = train_test_split(claims_data_encoded[payer_cat][state],claims_labels_dict[payer_cat][state][label], test_size=TEST_FRACTION, random_state=42)
                        print(payer_cat, state, label, "train:", train_set[payer_cat][state][label].shape, "test:", test_set[payer_cat][state][label].shape,"train_label:", y_train[payer_cat][state][label].shape, "test_label:", y_test[payer_cat][state][label].shape )
                    except :
                        print("There must be no data for payer category, state, and label combo :", payer_cat, state, label)

    return train_set, test_set, y_train, y_test


def prepare_train_label_data_full(claims_data_encoded, claims_labels_dict):
    dict_labels = DF_cat_labels
    dict_labels.append(NUMERIC)
    train_set = {}
    test_set = {}
    y_train = {}
    y_test = {}
    for label in dict_labels: 
        print(label)
        try:
            train_set[label], test_set[label],y_train[label], y_test[label] = train_test_split(claims_data_encoded,claims_labels_dict[label], test_size=TEST_FRACTION, random_state=42)
            print( label, "train:", train_set[label].shape, "test:", test_set[label].shape,"train_label:", y_train[label].shape, "test_label:", y_test[label].shape )
        except :
            print("There must be no data for label:", label)
    return train_set, test_set, y_train, y_test

def evaluate(model, train_set, y_train, test_set, y_test, label, label_index, data_fraction, measure_mse):
    mean_sqr_err = {} 
    score={}
    ascore={}
    y_predict_hat = {} 
    i = 0
    if (data_fraction <= 0 or data_fraction > 1):
        data_fraction = 1
        
    for payer_cat in payer_cat_value_count.index:
        score[payer_cat] = {}
        ascore[payer_cat] = {}
        mean_sqr_err[payer_cat] = {}
        y_predict_hat[payer_cat] = {}

        for state in state_value_count.index:
            if state != 0:
                train_data_set, train_label_set, test_data_set, test_label_set = get_shaped_data(train_set[payer_cat][state], y_train[payer_cat][state], test_set[payer_cat][state], y_test[payer_cat][state], label, label_index, data_fraction)
                try:
                    startTime = time.time()
                    score[payer_cat][state] = []
                    ascore[payer_cat][state] = []
                    mean_sqr_err[payer_cat][state] = []
                    y_predict_hat[payer_cat][state] = []
                    model.fit(train_data_set, train_label_set)
                    y_predict_hat[payer_cat][state] = model.predict(test_data_set)
                    score[payer_cat][state] = model.score(test_data_set, test_label_set)
                    #ascore[payer_cat][state] = accuracy_score(test_label[0:m1,label_index], y_predict_hat[payer_cat][state][0:m1])
                    if (measure_mse == True):
                        mean_sqr_err[payer_cat][state] = mean_squared_error(test_label_set, y_predict_hat[payer_cat][state])
                    endTime = time.time()
                    i = i + 1
                    print(payer_cat, state, 'ascore:', ascore[payer_cat][state], 'score:', score[payer_cat][state], 'Mean Error:', np.sqrt(mean_sqr_err[payer_cat][state]), 'time:', endTime - startTime, 'Dataset Shape:')
                    value, counts = np.unique(y_predict_hat[payer_cat][state], return_counts= True)
                    value1, counts1 = np.unique(test_label_set, return_counts= True)
                    print("Test Set Predicted: ", value, counts, "Actual: ", value1, counts1)
                except:    
                    print('Skipping: ', state)
    return score, y_predict_hat, mean_sqr_err

def get_shaped_data(train_set, y_train, test_set, y_test, label, label_index, data_fraction):
    if (data_fraction <= 0 or data_fraction > 1):
        data_fraction = 1
        
    M,N = train_set[label].shape
    M1,N1 = test_set[label].shape
    try:
        y_train[label].shape
        train_label = y_train[label]
        test_label = y_test[label]
    except:
        train_label = y_train[label].reshape(-1,1)
        test_label = y_test[label].reshape(-1,1)
                    
    m = np.floor(M * data_fraction).astype(int)
    m1 = np.floor(M1 * data_fraction).astype(int)
    train_data_set = train_set[label][0:m, :]
    train_label_set = train_label[0:m,label_index]
    test_data_set = test_set[label][0:m1]
    test_label_set = test_label[0:m1,label_index]
    
    return train_data_set, train_label_set, test_data_set, test_label_set
        
def evaluate_full(model, train_set, y_train, test_set, y_test, label, label_index, data_fraction, measure_mse):
    mean_sqr_err = {} 
    score={}
    y_predict_hat = {} 
    train_data_set, train_label_set, test_data_set, test_label_set = get_shaped_data(train_set, y_train, test_set, y_test, label, label_index, data_fraction)
    i = 0
    try:
        startTime = time.time()
        score = []
        mean_sqr_err = []
        y_predict_hat = []
        model.fit(train_data_set, train_label_set)
        endTime = time.time()
        print('Done with fitting:' ,endTime - startTime)
        y_predict_hat = model.predict(test_data_set)
        print('Done with predicting:', endTime-startTime)
        score = model.score(test_data_set, test_label_set)
        #ascore = accuracy_score(test_label[0:m1,label_index], y_predict_hat[0:m1])
        if (measure_mse == True):
            mean_sqr_err = mean_squared_error(test_label_set, y_predict_hat)
        endTime = time.time()
        i = i + 1
        print( 'score:', score, 'Mean Error:', np.sqrt(mean_sqr_err), 'time:', endTime - startTime, 'Dataset Shape:')
        value, counts = np.unique(y_predict_hat, return_counts= True)
        value1, counts1 = np.unique(test_label_set, return_counts= True)
        print("Test Set Predicted: ", value, counts, "Actual: ", value1, counts1)
    except Exception as e:    
        print(traceback.format_exception(*sys.exc_info()))
    return score, y_predict_hat, mean_sqr_err
	
	DF_cat_features = ['CPT', 'Mod1', 'Mod2','Mod3', 'Mod4', 'ICD1', 'ICD2', 'ICD3', 'ICD4', 
                   'PayerCategory','PayerState','Specialty', 'Sex', 'InsuranceType']
                
  
  
                
DF_cat_labels = ['ClaimStatus','R1', 'R2', 'R3', 'R4',
                'AdjCode1', 'AdjCode2', 'AdjCode3', 'AdjCode4', 'AdjCode5']
def prepare_one_hot_encoded_data_full(df):

    #now create a dataframe for each state before proceeding

    #cleanup and scale the numerical data 
    num_feature_pipeline = Pipeline([
        ('selector', DataFrameNumericCleanerSelector(DF_numeric_features)),
        ('std_scaler', StandardScaler())
        ])
    num_label_pipeline = Pipeline([
        ('selector', DataFrameNumericCleanerSelector(DF_numeric_labels)),
        ])
    claims_labels_dict = {}
    numeric_features = num_feature_pipeline.fit_transform(df)
    claims_feature_encoded_coo = coo_matrix(arg1=numeric_features)
    #create a dictionary so that the data can be grouped based on various labels
    claims_labels_dict[NUMERIC] = num_label_pipeline.fit_transform(df)
    
    #one hot encode the categorial labels and features
    for category in DF_cat_labels:
        selector = DataFrameSelector(category)
        claims_labels_dict[category] = get_one_hot_category_data(selector, df)
        print('label shape: ', category, claims_labels_dict[category].shape)
    for category in DF_cat_features:
        selector = DataFrameSelector(category)
        temp = get_one_hot_category_data(selector,df);
        print('feature shape: ',category,  temp.shape)
        claims_feature_encoded_coo = sparse.hstack([claims_feature_encoded_coo,temp])

    selector = DataFrameSelectorAsCategory(FACILITY_TYPE)
    claims_data_encoded_coo = sparse.hstack([claims_feature_encoded_coo,get_one_hot_category_data(selector, df)])

    return claims_data_encoded_coo, claims_labels_dict
claims_data_encoded_coo, claims_labels_dict = prepare_one_hot_encoded_data_full(df)
train_set, test_set, y_train, y_test = prepare_train_label_data_full(claims_data_encoded_coo, claims_labels_dict)    

tf.reset_default_graph()
label = 'ClaimStatus'
n_hidden_1 = 70
n_hidden_2 = 4
n_outputs = y_train[label].shape[1]
n_inputs = train_set[label].shape[1]
#X = tf.sparse_placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None,2), name= "y")
#one_hot_y = tf.placeholder(tf.int32, shape=(None, n_outputs), name= "one_hot_y")
sp_shape = tf.placeholder(tf.int64)
sp_ids = tf.placeholder(tf.int64)
sp_vals = tf.placeholder(tf.float32)
X = tf.SparseTensor(sp_ids, sp_vals, sp_shape)
def fetch_batch(X, y, epoch, batch_index, batch_size):
    start_index = batch_index*batch_size
    end_index = start_index + batch_size
    X_batch = X[start_index:end_index, :]
    y_batch = y[start_index:end_index, :]
    #tf.cast(X_batch,tf.float32)
    #tf.cast(y_batch,tf.float32)
    return X_batch, y_batch
def neuron_layer(X, n_neurons, n_inputs, name, activation=None):
    with tf.name_scope(name):
        stddev = 2 /np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal(shape=(n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name = "kernel")
        b = tf.Variable(tf.zeros(shape=n_neurons), name = "bias")
        Z = tf.matmul(X,W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

def neuron_layer_sparse(X, n_neurons, n_inputs, name, activation=None):
    with tf.name_scope(name):
        stddev = 2 /np.sqrt(n_inputs + n_neurons)
        init = tf.truncated_normal(shape=(n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name = "kernel")
        b = tf.Variable(tf.zeros(shape=n_neurons), name = "bias")
        Z = tf.sparse_tensor_dense_matmul(X,W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z
    
with tf.name_scope("dnn"):
    hidden1 = neuron_layer_sparse(X, n_hidden_1, n_inputs, name = "hidden1", activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden_2,n_hidden_1,  name = "hidden2", activation=tf.nn.relu)
    logits =  neuron_layer(hidden2, n_outputs,n_hidden_2,  name = "outputs")
    
    
#Now need a cost function, the xross entropy cost function -ylog(p) - (1-y)log(1-p)
#use coss_entropy_with_logits, use sparse matrix for better memory efficiency

with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits, name="crossEntropy")
    loss=tf.reduce_mean(xentropy, name="loss")


learning_rate = 0.1
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    
with tf.name_scope("eval"):
    #correct = tf.nn.in_top_k(logits, y, 1)
    correct = tf.equal(tf.argmax(y,1), tf.argmax(logits,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()



def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    print('Coo Shape:' , coo.shape)
    return tf.SparseTensor(indices, coo.data, coo.shape)
def fetch_batch(X, y, epoch, batch_index, batch_size):
    start_index = batch_index*batch_size
    end_index = start_index + batch_size
    X_batch =  (X[start_index:end_index, :])
    y_batch = (y[start_index:end_index, :].toarray())
    #X_batch = tf.cast(X_batch,tf.float32)
    #tf.cast(y_batch,tf.float32)
    return X_batch, y_batch
def get_shaped_data(train_set, y_train, test_set, y_test, label, label_index, data_fraction):
    if (data_fraction <= 0 or data_fraction > 1):
        data_fraction = 1
        
    M,N = train_set[label].shape
    M1,N1 = test_set[label].shape
    try:
        y_train[label].shape
        train_label = y_train[label]
        test_label = y_test[label]
    except:
        train_label = y_train[label].reshape(-1,1)
        test_label = y_test[label].reshape(-1,1)
                    
    m = np.floor(M * data_fraction).astype(int)
    m1 = np.floor(M1 * data_fraction).astype(int)
    train_data_set = train_set[label][0:m, :]
    train_label_set = train_label[0:m,label_index:]
    test_data_set = test_set[label][0:m1]
    test_label_set = test_label[0:m1,label_index:]
    
    return train_data_set, train_label_set, test_data_set, test_label_set



n_epochs = 20
batch_size = 50000
now = time.strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
log_dir = "{}/run-{}/".format(root_logdir, now)
loss_summary = tf.summary.scalar('XEntropy', loss)
file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
X, ylabels, test_X, test_y = get_shaped_data(train_set, y_train,test_set, y_test,label, 0, 1)
n_batches = int(np.floor(X.shape[0]/batch_size))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(X, ylabels, epoch, batch_index, batch_size)
            coo = X_batch.tocoo()
            indices = np.mat([coo.row, coo.col]).transpose()
            if (batch_index%10 == 0):
                summary_str = loss_summary.eval(feed_dict={sp_ids:indices,sp_vals:coo.data, sp_shape:coo.shape, y:y_batch})                                        
                step = epoch * n_batches + batch_index 
                file_writer.add_summary(summary_str, step)
            try:
                sess.run(training_op, feed_dict={sp_ids:indices,sp_vals:coo.data, sp_shape:coo.shape, y:y_batch})
            except:
                print("Errored_batch_index= ", batch_index)
            
            acc_train = accuracy.eval(feed_dict={sp_ids:indices,sp_vals:coo.data, sp_shape:coo.shape, y:y_batch}) 
            val_coo = test_X.tocoo()
            val_indices = np.mat([val_coo.row, val_coo.col]).transpose()
            acc_val = accuracy.eval(feed_dict={sp_ids:val_indices,sp_vals:val_coo.data, sp_shape:val_coo.shape, y:test_y.toarray()})   
        
        print(epoch, "train accuracy: " , acc_train , "val accuracy:", acc_val)    
        if (epoch % 5 == 0):
            print ("Epoch", epoch, "loss = ", loss.eval(feed_dict={sp_ids:indices,sp_vals:coo.data, sp_shape:coo.shape, y:y_batch}))
    save_path = saver.save(sess, "./70-4claimsdnn.ckpt")
    file_writer.close()