# %% [markdown]
# ## SVM execution on feature extraction from the dataset

# %%
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,scale
from sklearn.model_selection import GridSearchCV, cross_validate,train_test_split,KFold
import json
from sklearn.metrics import classification_report,ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import pickle
import copy
pd.set_option("display.max_columns", None)

# %%
df_action40_train = pd.read_csv('dt_avg_resnet_optm_train.csv')
df_action40_test = pd.read_csv('dt_avg_resnet_optm_test.csv')

'''
df_cifar10_train = pd.read_csv('dt_avg_resnet_cifar10_optm_train.csv')
df_cifar10_test = pd.read_csv('dt_avg_resnet_cifar10_optm_test.csv')

df_cifar100_train = pd.read_csv('df_resnet_final_last_cifar100_train.csv')
df_cifar100_test = pd.read_csv('df_resnet_final_last_cifar100_test.csv')

df_cub200_train = pd.read_csv('df_resnet_final_last_cub200_train.csv')
df_cub200_test = pd.read_csv('df_resnet_final_last_cub200_test.csv')
'''

# %% [markdown]
# Defining X and y

# %%
df_action40_train.drop(columns=['Unnamed: 0'],inplace=True)
df_action40_test.drop(columns=['Unnamed: 0'],inplace=True)

# %%
df_action40_train.shape

# %%
X_train = df_action40_train.drop(['class','name'],axis=1)
y_train = df_action40_train['class']

X_test = df_action40_test.drop(['class','name'],axis=1)
y_test = df_action40_test['class']

# %% [markdown]
# Get the best hyperparameters

# %%
param_grid = {
    'C': [1,0.01, 0.1],  
    'penalty': ['l1','l2'], 
    'loss': ['hinge', 'squared_hinge'],
    #'dual': [True, False], #number of samples < number of features
    #'multi_class': ['crammer_singer','ovr'], #crammer too expensive
    #'class_weight': ['balanced', None], #is balanced
    'max_iter': [1000, 2000, 100],
    'random_state':[7]
    }

svm=LinearSVC()
svm_cv=GridSearchCV(svm,param_grid,cv=5)
svm_cv.fit(X_train, y_train)

print("best parameters",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)

# %% [markdown]
# SUM:
# best parameters {'C': 0.001, 'class_weight': 'balanced', 'dual': True, 'loss': 'hinge', 'max_iter': 1000, 'multi_class': 'crammer_singer', 'penalty': 1}
# accuracy : 0.7801271274475121

# %% [markdown]
# best parameters {'C': 0.01, 'class_weight': 'balanced', 'dual': True, 'loss': 'hinge', 'max_iter': 1000, 'multi_class': 'crammer_singer', 'penalty': 1}
# accuracy : 0.5751774674460524

# %% [markdown]
# Training the classifier

# %%
clf = LinearSVC(**svm_cv.best_params_)

# %% [markdown]
# Running a cross-validation

# %%
def training_evaluation(X,y,pipe,k=5):
    acc=[]
    pre=[]
    rec=[]
    f1=[]
    
    better_model=0
    better_predi=0
    better_metric=0
    better_test=0
    qtd_class=0
    classes=[]
    
    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = copy.deepcopy(pipe)
        model.fit(X_train,y_train)
        predictions = model.predict(X_test)

        #calculating the metrics by each class
        acc.append(metrics.accuracy_score(y_test,predictions))
        pre.append(metrics.precision_score(y_test,predictions,average='macro'))
        rec.append(metrics.recall_score(y_test,predictions,average='macro'))
        f1.append(metrics.f1_score(y_test,predictions,average='macro'))
        classes = model.classes_

        if f1[-1] > better_metric:
            better_metric = f1[-1]
            better_model = copy.deepcopy(model)#model
            better_pred = model.predict(X_test)
            better_test = y_test.copy()
            better_x_test = X_test.copy()

    #plotting the metrics for the kfold
    fig, ax = plt.subplots(figsize=(10,10)) 
    width = 0.2
    r = 4
    n=np.arange(r)

    plt.bar(n+width, height=[np.mean(pre),np.mean(rec),np.mean(f1),np.mean(acc)],
           yerr=[np.std(pre),np.std(rec),np.std(f1),np.std(acc)])

    plt.title('Model Evaluation')
    plt.xticks(np.arange(r),['Precision','Recall','F1','Accuracy'],rotation=90)
    plt.legend(bbox_to_anchor=(1.20,1))
    plt.grid()
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()

    return better_model, better_pred, better_test, better_x_test

# %%
model,pred,y_test,X_test = training_evaluation(pd.concat([X_train,X_test]),pd.concat([y_train,y_test]),clf)

# %% [markdown]
# save the model

# %%
clf = copy.deepcopy(model)
y_test = y_test
y_pred = pred
X_test = X_test

# %%
pickle.dump(clf, open(f'svm_action40.pkl', 'wb'))
pickle.dump(y_test, open(f'svm_y_test_action40.pkl', 'wb'))
pickle.dump(y_pred, open(f'svm_y_pred_action40.pkl', 'wb'))
pickle.dump(X_test, open(f'svm_X_test_action40.pkl', 'wb'))

# %%
print(classification_report(y_test, y_pred, target_names=clf.classes_))

# %% [markdown]
# confusion matrix

# %%
fig, ax = plt.subplots(figsize=(20, 20))
plot_confusion_matrix(clf, X_test, y_test, ax=ax,normalize='true',values_format='.0%')  
plt.xticks(rotation=90,fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('True Label',fontsize=20)
plt.xlabel('Predicted Label',fontsize=20)
plt.show()  

# %%
feature_importance_dict={}

for clas,coef in zip(clf.classes_,clf.coef_):
    feature_importance_dict[clas]=coef

# %% [markdown]
# #### Export files with ranked features per class

# %%
positive={}
negative={}
for i,j in enumerate(clf.classes_):
    coef = clf.coef_[i].ravel()
    p=[]
    for k in np.argsort(coef)[-10:]:
        p.append(k+1)
    positive[j]= p

# %%
positive_df = pd.DataFrame.from_dict(positive,orient='index').rename(
    columns={0:'9',1:'8',2:'7',3:'6',4:'5',5:'4',6:'3',7:'2',8:'1',9:'0'}).stack()

# %%
positive_df = pd.DataFrame(positive_df).rename(columns={0:'unit'}).reset_index().rename(
    columns={'level_0':'class','level_1':'unit_rank'})

# %% [markdown]
# Join with the result of netdissection

# %%
net_result = pd.read_csv(f'result/pytorch_{settings.MODEL}_{settings.DATASET}/tally.csv')

# %%
positive_net=positive_df.merge(net_result,on='unit',how='inner')

# %% [markdown]
# Selecting unique features

# %%
positive_net.unit_rank = positive_net.unit_rank.astype(np.int16)

# %%
pos_unique = positive_net.sort_values(['class','unit_rank']).drop_duplicates(['class','label']).groupby(['class']).head(10)

# %%
p_unique = {}
for p in pos_unique.values:
    if p[0] in p_unique:
        if p[3] in p_unique[p[0]]:
            p_unique[p[0]][p[3]].append(p[4])
        else:
            p_unique[p[0]][p[3]]= [p[4]]
    else:
        p_unique[p[0]]= {p[3]:[p[4]]}
        
n_unique = {}
for n in neg_unique.values:
    if n[0] in n_unique:
        if n[3] in n_unique[n[0]]:
            n_unique[n[0]][n[3]].append(n[4])
        else:
            n_unique[n[0]][n[3]]= [n[4]]
    else:
        n_unique[n[0]]= {n[3]:[n[4]]}

# %%
pickle.dump(p_unique,open('global_positive_features_svm.pkl','wb'))


