import pandas as pd #Load necessary packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn import tree

cancer_wisco = pd.read_csv("C:/Users/drone/Downloads/archive/wisconsin_b_cancer.csv")
print(cancer_wisco.head()) #Load data set

print(cancer_wisco.isnull().sum()) #Check null values

wisco2=cancer_wisco.drop(["Unnamed: 32", "id"],axis=1) #Drop column with null values, drop id (not necessary for model)
print(wisco2.head())

features_mean= list(wisco2.columns[1:11])
features_se= list(wisco2.columns[11:20])
features_worst=list(wisco2.columns[21:31])

corr = wisco2[features_mean].corr()
sns.heatmap(corr, annot=True, linewidth=.5, fmt=".1f", cmap= 'coolwarm')
plt.show()

wisco2['diagnosis']=wisco2['diagnosis'].map({'M':1,'B':0})
train, test = train_test_split(wisco2, test_size=0.3, random_state=65)
print(train.shape)
print(test.shape)
X_train = train[features_mean]
X_test = test[features_mean]
y_train = train.diagnosis
y_test = test.diagnosis

wisco2_model1 = RandomForestClassifier(n_estimators=50, random_state=65)
wisco2_model2 = RandomForestClassifier(n_estimators=100, random_state=65)
wisco2_model3 = RandomForestClassifier(n_estimators=150, random_state=65)
wisco2_model4 = RandomForestClassifier(n_estimators=400, random_state=65)
wisco2_model5 = RandomForestClassifier(n_estimators=50, min_samples_split=20, random_state=65)
wisco2_model6 = RandomForestClassifier(n_estimators=100, min_samples_split=20, random_state=65)
wisco2_model7 = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=0)

models = [wisco2_model1, wisco2_model2, wisco2_model3, wisco2_model4, wisco2_model5, wisco2_model6, wisco2_model7]

def score_model(model, X_t=X_train, X_v=X_test, y_t=y_train, y_v=y_test):
   model.fit(X_t, y_t)
   preds = model.predict(X_v)
   return accuracy_score(preds, y_v)

for x in models:
    print(score_model(x))

importance = wisco2_model3.feature_importances_
columns = wisco2[features_mean].columns
i = 0

while i < len(columns):
    print(f"The importance of feature {columns[i]} is {round(importance[i] * 100, 2)}%")
    i += 1

wisco2_smallMod = RandomForestClassifier(n_estimators=10, max_depth = 4)
wisco2_smallMod.fit(X_train, y_train)
wisco2_small = wisco2_smallMod.estimators_[5]
feature_names = wisco2[features_mean].columns
wisco2['diagnosis']=wisco2['diagnosis'].map({1: "M", 0: "B"})
target_names = wisco2['diagnosis'].unique().tolist()
plt.figure(figsize=(12,12))  # set plot size (denoted in inches)
plot_tree(wisco2_small, 
                                feature_names= feature_names,  
                                class_names= target_names,
                                filled=True, fontsize = 8)
plt.show()

wisco2['diagnosis']=wisco2['diagnosis'].map({'M':1,'B':0})

y_pred = wisco2_model3.predict(X_test)
comparison_df = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(comparison_df)

plt.figure(figsize=(5, 7))

fig, ax = plt.subplots()
sns.kdeplot(wisco2['diagnosis'], ax=ax)
sns.kdeplot(y_pred, ax=ax)
plt.legend(labels=["Actual", "Predicted"])

plt.title('Actual vs Fitted Values for Diagnosis')

plt.show()
plt.close()