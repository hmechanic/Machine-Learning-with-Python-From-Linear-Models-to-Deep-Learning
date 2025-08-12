import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    
    svm_model = LinearSVC(random_state=0 , C=1)
    svm_model.fit(train_x, train_y)
    y_pred = svm_model.predict(test_x)
    
    return y_pred


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    
    unique_y_values = np.unique(train_y)
    models = []
    for i in unique_y_values:
        
        train_y_binary  = (train_y == i).astype(int)
        
        svm_model = LinearSVC(random_state=0 , C=0.1)
        svm_model.fit(train_x, train_y_binary)
        models.append(svm_model)
       
    y_predict_all_models = []
    
    for model in models:
        
        y_pred_i = model.decision_function(test_x)
        y_predict_all_models.append(y_pred_i)
        
    y_predict_all_models = np.array(y_predict_all_models)
    pred_test_y = y_predict_all_models.argmax(axis=0)
    
    
    ### Course implementation
    
    # clf = LinearSVC(C=0.1, random_state=0)
    # clf.fit(train_x, train_y)
    # pred_test_y = clf.predict(test_x)
    
    return pred_test_y
        
def compute_test_error_svm(test_y, pred_test_y):
    accuracy = accuracy_score(test_y, pred_test_y)
    return accuracy