#!/usr/bin/env python
# coding: utf-8

# In[1]:


def plot_coefficients(clf):
    weights_clf = pd.Series(clf.coef_[0], index=scaled_X_train_log.columns.values)
    weights_clf.sort_values(inplace=True)
    plt.figure(figsize=(20, 6))
    plt.xticks(rotation=90)
    features = plt.bar(weights_clf.index, weights_clf.values)
    
    
    
def model_evaluation(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test):

    print('MODEL EVALUATION METRICS:\n',
          '-----------------------------------------------------')

    print('Confusion Matrix for train & test set: \n',
          '\nTrain set\n',
          confusion_matrix(y_train, y_pred_train), '\n'
          '\n\nTest set\n',)
    print(confusion_matrix(y_train, y_pred_train), '\n')
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


# In[ ]:




