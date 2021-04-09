#!/usr/bin/env python
# coding: utf-8

# In[2]:


def plot_coefficients(clf):
    """
        The function to plot the coefficients of logistic regression model.
  
        Parameters:
            clf: classifier model   
    """
    
    weights_clf = pd.Series(clf.coef_[0], index=scaled_X_train_log.columns.values)
    weights_clf.sort_values(inplace=True)
    plt.figure(figsize=(20, 6))
    plt.xticks(rotation=90)
    #barplot
    features = plt.bar(weights_clf.index, weights_clf.values)
    
    
    
def model_evaluation(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test):
    """
        The function to print the metrics of the model evaluation.
        Printed metrics: Accuracy scores, 
                         confusion matrixes, 
                         and classification reports for train and test sets, 
  
        Parameters: Train, test, and prediction values
                    X_train, X_test, 
                    y_train, y_test, 
                    y_pred_train, y_pred_test   
    """

    print('MODEL EVALUATION METRICS:\n',
          '-----------------------------------------------------')
    print('Train Set Accuracy Score', round(accuracy_score(y_train, y_pred_train), 6))
    print('Test Set Accuracy Score', round(accuracy_score(y_test, y_pred_test),6))
    print('-----------------------------------------------------\n')

    print('Confusion Matrix for train & test set: \n',
          '\nTrain set\n',
          confusion_matrix(y_train, y_pred_train), '\n'
          '\n\nTest set\n',)
    print(confusion_matrix(y_test, y_pred_test), '\n')


    print('-----------------------------------------------------')
    print('\nClassification Report for train & test set\n',
          '\nTrain set\n',
          classification_report(y_train, y_pred_train),
          '\n\nTest set\n',
          classification_report(y_test, y_pred_test))

    print('-----------------------------------------------------\n')

    # print('roc auc score for train and test set:\n ',
    #       round(roc_auc_score(y_train, y_pred_train), 4),
    #       round(roc_auc_score(y_test, y_pred_test), 4))
    
    
def plot_feature_importances(model):
    """
        The function to plot the coefficients of tree based models.
  
        Parameters:
            model: classifier model  
    """
    n_features = X_train.shape[1]
    plt.figure(figsize=(20, 15))
    #barplot
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns.values)
    plt.title('Comparison of Feature Importances')
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    


# In[ ]:




