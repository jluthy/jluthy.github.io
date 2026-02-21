---
layout: default
modal-id: 1
date: 2014-07-18
img: cabin.png
alt: image-alt
project-date: April 2014
client: Start Bootstrap
category: Web Development
description: Use this area of the page to describe your project. Lorem ipsum dolor sit amet, consectetur adipisicing elit. Mollitia neque assumenda ipsam nihil, molestias magnam, recusandae quos quis inventore quisquam velit asperiores, vitae? Reprehenderit soluta, eos quod consequuntur itaque. Nam.
---

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

# load in the data using pandas
data = pd.read_csv('The_Cancer_data_1500_V2.csv')

# Interaction Terms
data['Age_BMI'] = data['Age'] * data['BMI']
data['Age_PhysicalActivity'] = data['Age'] * data['PhysicalActivity']
data['BMI_PhysicalActivity'] = data['BMI'] * data['PhysicalActivity']

# Polynomial Features
# poly = PolynomialFeatures(degree=2, include_bias=False)
# poly_features = poly.fit_transform(data[['Age', 'BMI']])
# poly_feature_names = poly.get_feature_names_out(['Age', 'BMI'])
# poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
# data = pd.concat([data, poly_df], axis=1)

# Binning
# data['Age_binned'] = pd.cut(data['Age'], bins=[0, 30, 50, 70, 100], labels=['Young', 'Middle-aged', 'Senior', 'Elder'])
# data['BMI_binned'] = pd.cut(data['BMI'], bins=[0, 18.5, 25, 30, 40, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese', 'Severely Obese'])

# # Log Transformation
# data['AlcoholIntake_log'] = np.log1p(data['AlcoholIntake'])

# # Combine Categorical Features
# data['Gender_Smoking'] = data['Gender'].astype(str) + '_' + data['Smoking'].astype(str)

# PCA
# pca = PCA(n_components=5)
# pca_features = pca.fit_transform(data.drop('Diagnosis', axis=1))
# pca_feature_names = [f'PCA_{i}' for i in range(1, 6)]
# pca_df = pd.DataFrame(pca_features, columns=pca_feature_names)
# data = pd.concat([data, pca_df], axis=1)

print(data.head())count

data
```

       Age  Gender        BMI  Smoking  GeneticRisk  PhysicalActivity  \
    0   58       1  16.085313        0            1          8.146251   
    1   71       0  30.828784        0            1          9.361630   
    2   48       1  38.785084        0            2          5.135179   
    3   34       0  30.040296        0            0          9.502792   
    4   62       1  35.479721        0            0          5.356890   
    
       AlcoholIntake  CancerHistory  Diagnosis      Age_BMI  Age_PhysicalActivity  \
    0       4.148219              1          1   932.948173            472.482532   
    1       3.519683              0          0  2188.843692            664.675760   
    2       4.728368              0          1  1861.684011            246.488576   
    3       2.044636              0          0  1021.370047            323.094936   
    4       3.309849              0          1  2199.742732            332.127162   
    
       BMI_PhysicalActivity  
    0            131.034993  
    1            288.607686  
    2            199.168334  
    3            285.466687  
    4            190.060955  





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Gender</th>
      <th>BMI</th>
      <th>Smoking</th>
      <th>GeneticRisk</th>
      <th>PhysicalActivity</th>
      <th>AlcoholIntake</th>
      <th>CancerHistory</th>
      <th>Diagnosis</th>
      <th>Age_BMI</th>
      <th>Age_PhysicalActivity</th>
      <th>BMI_PhysicalActivity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>58</td>
      <td>1</td>
      <td>16.085313</td>
      <td>0</td>
      <td>1</td>
      <td>8.146251</td>
      <td>4.148219</td>
      <td>1</td>
      <td>1</td>
      <td>932.948173</td>
      <td>472.482532</td>
      <td>131.034993</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71</td>
      <td>0</td>
      <td>30.828784</td>
      <td>0</td>
      <td>1</td>
      <td>9.361630</td>
      <td>3.519683</td>
      <td>0</td>
      <td>0</td>
      <td>2188.843692</td>
      <td>664.675760</td>
      <td>288.607686</td>
    </tr>
    <tr>
      <th>2</th>
      <td>48</td>
      <td>1</td>
      <td>38.785084</td>
      <td>0</td>
      <td>2</td>
      <td>5.135179</td>
      <td>4.728368</td>
      <td>0</td>
      <td>1</td>
      <td>1861.684011</td>
      <td>246.488576</td>
      <td>199.168334</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34</td>
      <td>0</td>
      <td>30.040296</td>
      <td>0</td>
      <td>0</td>
      <td>9.502792</td>
      <td>2.044636</td>
      <td>0</td>
      <td>0</td>
      <td>1021.370047</td>
      <td>323.094936</td>
      <td>285.466687</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62</td>
      <td>1</td>
      <td>35.479721</td>
      <td>0</td>
      <td>0</td>
      <td>5.356890</td>
      <td>3.309849</td>
      <td>0</td>
      <td>1</td>
      <td>2199.742732</td>
      <td>332.127162</td>
      <td>190.060955</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1495</th>
      <td>62</td>
      <td>1</td>
      <td>25.090025</td>
      <td>0</td>
      <td>0</td>
      <td>9.892167</td>
      <td>1.284158</td>
      <td>0</td>
      <td>1</td>
      <td>1555.581554</td>
      <td>613.314344</td>
      <td>248.194714</td>
    </tr>
    <tr>
      <th>1496</th>
      <td>31</td>
      <td>0</td>
      <td>33.447125</td>
      <td>0</td>
      <td>1</td>
      <td>1.668297</td>
      <td>2.280636</td>
      <td>1</td>
      <td>1</td>
      <td>1036.860863</td>
      <td>51.717209</td>
      <td>55.799739</td>
    </tr>
    <tr>
      <th>1497</th>
      <td>63</td>
      <td>1</td>
      <td>32.613861</td>
      <td>1</td>
      <td>1</td>
      <td>0.466848</td>
      <td>0.150101</td>
      <td>0</td>
      <td>1</td>
      <td>2054.673224</td>
      <td>29.411437</td>
      <td>15.225722</td>
    </tr>
    <tr>
      <th>1498</th>
      <td>55</td>
      <td>0</td>
      <td>25.568216</td>
      <td>0</td>
      <td>0</td>
      <td>7.795317</td>
      <td>1.986138</td>
      <td>1</td>
      <td>1</td>
      <td>1406.251876</td>
      <td>428.742425</td>
      <td>199.312344</td>
    </tr>
    <tr>
      <th>1499</th>
      <td>67</td>
      <td>1</td>
      <td>23.663104</td>
      <td>0</td>
      <td>0</td>
      <td>2.525860</td>
      <td>2.856600</td>
      <td>1</td>
      <td>0</td>
      <td>1585.427981</td>
      <td>169.232624</td>
      <td>59.769690</td>
    </tr>
  </tbody>
</table>
<p>1500 rows × 12 columns</p>
</div>




```python
# Inspect the data further to see the data types and non-null counts
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1500 entries, 0 to 1499
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   Age                   1500 non-null   int64  
     1   Gender                1500 non-null   int64  
     2   BMI                   1500 non-null   float64
     3   Smoking               1500 non-null   int64  
     4   GeneticRisk           1500 non-null   int64  
     5   PhysicalActivity      1500 non-null   float64
     6   AlcoholIntake         1500 non-null   float64
     7   CancerHistory         1500 non-null   int64  
     8   Diagnosis             1500 non-null   int64  
     9   Age_BMI               1500 non-null   float64
     10  Age_PhysicalActivity  1500 non-null   float64
     11  BMI_PhysicalActivity  1500 non-null   float64
    dtypes: float64(6), int64(6)
    memory usage: 140.8 KB



```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Define feature columns
categorical_columns = ['Gender', 'Smoking', 'CancerHistory', 'GeneticRisk']
numerical_columns = ['Age', 'BMI', 'PhysicalActivity', 'AlcoholIntake', 'Age_BMI', 'Age_PhysicalActivity', 'BMI_PhysicalActivity']

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Split data into training and test sets
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Further split the temporary set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Apply preprocessing
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_val_preprocessed = preprocessor.transform(X_val)
X_test_preprocessed = preprocessor.transform(X_test)

# Dictionary of classification models
classification_models = {
    "Logistic Regression": LogisticRegression(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(silent=True),
}

model_names = []
accuracies = []
roc_auc_scores = []

# Train and evaluate each model
for name, clf in classification_models.items():
    clf.fit(X_train_preprocessed, y_train)
    y_test_pred = clf.predict(X_test_preprocessed)
    y_test_pred_prob = clf.predict_proba(X_test_preprocessed)[:, 1]
    accuracy = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_pred_prob)
    
    model_names.append(name)
    accuracies.append(accuracy)
    roc_auc_scores.append(roc_auc)
    
    print(f"{name} accuracy: {accuracy:.2f}")
    print(f"{name} ROC AUC Score: {roc_auc:.2f}")
    print(f"{name} Classification Report:\n{classification_report(y_test, y_test_pred)}")
    
# Create a DataFrame for model accuracies
df = pd.DataFrame({'Model': model_names, 
                   'Accuracy': accuracies,
                   'ROC AUC Score': roc_auc_scores})

print(df)



# Plot model accuracies using Seaborn
plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='Accuracy', data=df)
plt.title('Model Accuracies')
plt.xticks(rotation=45)
plt.show()

# Find the best model
best_index = accuracies.index(max(accuracies))
best_model_name = model_names[best_index]
best_model = classification_models[best_model_name]

print(f"The best model is: {best_model_name} with an accuracy of {accuracies[best_index]:.2f}")
```

    /Users/jluthy/miniconda3/envs/xgboosty/lib/python3.10/site-packages/xgboost/sklearn.py:1395: UserWarning: `use_label_encoder` is deprecated in 1.7.0.
      warnings.warn("`use_label_encoder` is deprecated in 1.7.0.")


    Logistic Regression accuracy: 0.89
    Logistic Regression ROC AUC Score: 0.95
    Logistic Regression Classification Report:
                  precision    recall  f1-score   support
    
               0       0.88      0.94      0.91       137
               1       0.90      0.81      0.85        88
    
        accuracy                           0.89       225
       macro avg       0.89      0.87      0.88       225
    weighted avg       0.89      0.89      0.89       225
    
    K-Nearest Neighbors accuracy: 0.84
    K-Nearest Neighbors ROC AUC Score: 0.91
    K-Nearest Neighbors Classification Report:
                  precision    recall  f1-score   support
    
               0       0.83      0.94      0.88       137
               1       0.88      0.69      0.78        88
    
        accuracy                           0.84       225
       macro avg       0.86      0.82      0.83       225
    weighted avg       0.85      0.84      0.84       225
    
    Support Vector Machine accuracy: 0.89
    Support Vector Machine ROC AUC Score: 0.95
    Support Vector Machine Classification Report:
                  precision    recall  f1-score   support
    
               0       0.89      0.93      0.91       137
               1       0.89      0.82      0.85        88
    
        accuracy                           0.89       225
       macro avg       0.89      0.88      0.88       225
    weighted avg       0.89      0.89      0.89       225
    
    Decision Tree accuracy: 0.85
    Decision Tree ROC AUC Score: 0.84
    Decision Tree Classification Report:
                  precision    recall  f1-score   support
    
               0       0.88      0.88      0.88       137
               1       0.81      0.81      0.81        88
    
        accuracy                           0.85       225
       macro avg       0.84      0.84      0.84       225
    weighted avg       0.85      0.85      0.85       225
    
    Random Forest accuracy: 0.91
    Random Forest ROC AUC Score: 0.95
    Random Forest Classification Report:
                  precision    recall  f1-score   support
    
               0       0.91      0.94      0.92       137
               1       0.90      0.85      0.88        88
    
        accuracy                           0.91       225
       macro avg       0.91      0.90      0.90       225
    weighted avg       0.91      0.91      0.91       225
    
    Gradient Boosting accuracy: 0.91
    Gradient Boosting ROC AUC Score: 0.97
    Gradient Boosting Classification Report:
                  precision    recall  f1-score   support
    
               0       0.90      0.96      0.93       137
               1       0.93      0.84      0.88        88
    
        accuracy                           0.91       225
       macro avg       0.91      0.90      0.91       225
    weighted avg       0.91      0.91      0.91       225
    
    AdaBoost accuracy: 0.91
    AdaBoost ROC AUC Score: 0.97
    AdaBoost Classification Report:
                  precision    recall  f1-score   support
    
               0       0.90      0.95      0.93       137
               1       0.91      0.84      0.88        88
    
        accuracy                           0.91       225
       macro avg       0.91      0.89      0.90       225
    weighted avg       0.91      0.91      0.91       225
    
    Gaussian Naive Bayes accuracy: 0.82
    Gaussian Naive Bayes ROC AUC Score: 0.93
    Gaussian Naive Bayes Classification Report:
                  precision    recall  f1-score   support
    
               0       0.80      0.93      0.86       137
               1       0.86      0.64      0.73        88
    
        accuracy                           0.82       225
       macro avg       0.83      0.79      0.80       225
    weighted avg       0.82      0.82      0.81       225
    
    XGBoost accuracy: 0.90
    XGBoost ROC AUC Score: 0.96
    XGBoost Classification Report:
                  precision    recall  f1-score   support
    
               0       0.89      0.95      0.92       137
               1       0.91      0.82      0.86        88
    
        accuracy                           0.90       225
       macro avg       0.90      0.88      0.89       225
    weighted avg       0.90      0.90      0.90       225
    
    CatBoost accuracy: 0.93
    CatBoost ROC AUC Score: 0.97
    CatBoost Classification Report:
                  precision    recall  f1-score   support
    
               0       0.94      0.96      0.95       137
               1       0.93      0.90      0.91        88
    
        accuracy                           0.93       225
       macro avg       0.93      0.93      0.93       225
    weighted avg       0.93      0.93      0.93       225
    
                        Model  Accuracy  ROC AUC Score
    0     Logistic Regression  0.888889       0.954794
    1     K-Nearest Neighbors  0.844444       0.913902
    2  Support Vector Machine  0.888889       0.952223
    3           Decision Tree  0.848889       0.841365
    4           Random Forest  0.906667       0.954919
    5       Gradient Boosting  0.911111       0.970886
    6                AdaBoost  0.906667       0.965246
    7    Gaussian Naive Bayes  0.817778       0.931652
    8                 XGBoost  0.897778       0.963006
    9                CatBoost  0.933333       0.973208



    
![png](output_2_2.png)
    


    The best model is: CatBoost with an accuracy of 0.93


The two best models were AdaBoost and CatBoost. I will attempt to tune each one further.



```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for AdaBoost
param_grid_adaboost = {
    'n_estimators': [79, 80, 81, 82, 83],
    'learning_rate': [0.19, 0.20, 0.21, 0.22]
}

# Initialize the AdaBoost classifier
adaboost = AdaBoostClassifier()

# Set up GridSearchCV for AdaBoost
grid_search_adaboost = GridSearchCV(estimator=adaboost, param_grid=param_grid_adaboost, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search_adaboost.fit(X_train_preprocessed, y_train)

# Get the best parameters and model for AdaBoost
best_params_adaboost = grid_search_adaboost.best_params_
best_model_adaboost = grid_search_adaboost.best_estimator_

print("Best parameters for AdaBoost:", best_params_adaboost)
print("Best AdaBoost model accuracy on validation data:", best_model_adaboost.score(X_val_preprocessed, y_val))
print("Best AdaBoost model ROC AUC on validation data:", roc_auc_score(y_val, best_model_adaboost.predict_proba(X_val_preprocessed)[:, 1]))

```

    Best parameters for AdaBoost: {'learning_rate': 0.19, 'n_estimators': 81}
    Best AdaBoost model accuracy on validation data: 0.9244444444444444
    Best AdaBoost model ROC AUC on validation data: 0.9314092953523239



```python
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for CatBoost
param_grid_catboost = {
    'iterations': [160, 161, 162, 163, 164, 165, 166, 167],
    'depth': [2, 3],
    'learning_rate': [0.11, 0.12, 0.13, 0.14, 0.15]
}

# Initialize the CatBoost classifier
catboost = CatBoostClassifier(silent=True)

# Set up GridSearchCV for CatBoost
grid_search_catboost = GridSearchCV(estimator=catboost, param_grid=param_grid_catboost, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search_catboost.fit(X_train_preprocessed, y_train)

# Get the best parameters and model for CatBoost
best_params_catboost = grid_search_catboost.best_params_
best_model_catboost = grid_search_catboost.best_estimator_

print("Best parameters for CatBoost:", best_params_catboost)
print("Best CatBoost model accuracy on validation data:", best_model_catboost.score(X_val_preprocessed, y_val))
print("Best CatBoost model ROC AUC on validation data:", roc_auc_score(y_val, best_model_catboost.predict_proba(X_val_preprocessed)[:, 1]))

```

    Best parameters for CatBoost: {'depth': 2, 'iterations': 162, 'learning_rate': 0.15}
    Best CatBoost model accuracy on validation data: 0.9377777777777778
    Best CatBoost model ROC AUC on validation data: 0.9315342328835583



```python
# Final evaluation on the test set for AdaBoost
y_test_pred_adaboost = best_model_adaboost.predict(X_test_preprocessed)
y_test_pred_prob_adaboost = best_model_adaboost.predict_proba(X_test_preprocessed)[:, 1]
print("AdaBoost Test ROC AUC Score:", roc_auc_score(y_test, y_test_pred_prob_adaboost))
print("AdaBoost Test Classification Report:\n", classification_report(y_test, y_test_pred_adaboost))

# Final evaluation on the test set for CatBoost
y_test_pred_catboost = best_model_catboost.predict(X_test_preprocessed)
y_test_pred_prob_catboost = best_model_catboost.predict_proba(X_test_preprocessed)[:, 1]
print("CatBoost Test ROC AUC Score:", roc_auc_score(y_test, y_test_pred_prob_catboost))
print("CatBoost Test Classification Report:\n", classification_report(y_test, y_test_pred_catboost))
```

    AdaBoost Test ROC AUC Score: 0.9786828135368282
    AdaBoost Test Classification Report:
                   precision    recall  f1-score   support
    
               0       0.91      1.00      0.95       137
               1       1.00      0.85      0.92        88
    
        accuracy                           0.94       225
       macro avg       0.96      0.93      0.94       225
    weighted avg       0.95      0.94      0.94       225
    
    CatBoost Test ROC AUC Score: 0.9803417385534174
    CatBoost Test Classification Report:
                   precision    recall  f1-score   support
    
               0       0.93      0.98      0.95       137
               1       0.96      0.89      0.92        88
    
        accuracy                           0.94       225
       macro avg       0.95      0.93      0.94       225
    weighted avg       0.94      0.94      0.94       225
    



```python
import matplotlib.pyplot as plt
import numpy as np

# Get feature importances from the best AdaBoost model
feature_importances_adaboost = best_model_adaboost.feature_importances_

# Get feature names from the preprocessor
numerical_features = numerical_columns
categorical_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_columns)
feature_names = np.concatenate([numerical_features, categorical_features])

# Create a DataFrame for easy plotting
feature_importance_df_adaboost = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances_adaboost
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df_adaboost['Feature'], feature_importance_df_adaboost['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - AdaBoost')
plt.gca().invert_yaxis()
plt.show()

```


    
![png](output_7_0.png)
    



```python
# Get feature importances from the best CatBoost model
feature_importances_catboost = best_model_catboost.get_feature_importance()

# Get feature names from the preprocessor
numerical_features = numerical_columns
categorical_features = preprocessor.transformers_[1][1]['onehot'].get_feature_names_out(categorical_columns)
feature_names = np.concatenate([numerical_features, categorical_features])

# Create a DataFrame for easy plotting
feature_importance_df_catboost = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances_catboost
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df_catboost['Feature'], feature_importance_df_catboost['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance - CatBoost')
plt.gca().invert_yaxis()
plt.show()

```


    
![png](output_8_0.png)
    



```python
print(feature_importance_df_catboost)
```

                     Feature  Importance
    15         GeneticRisk_2   14.664818
    11       CancerHistory_0   13.448981
    3          AlcoholIntake   12.370437
    1                    BMI   10.853871
    2       PhysicalActivity   10.040949
    0                    Age    8.115082
    8               Gender_1    6.866607
    10             Smoking_1    5.282098
    7               Gender_0    5.058323
    12       CancerHistory_1    4.924163
    4                Age_BMI    3.479967
    9              Smoking_0    3.044449
    6   BMI_PhysicalActivity    0.945496
    13         GeneticRisk_0    0.486859
    5   Age_PhysicalActivity    0.417898
    14         GeneticRisk_1    0.000000



```python
# Creation of confusion matrix for CatBoost
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_test_pred_catboost)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Cancer', 'Cancer'],
            yticklabels=['No Cancer', 'Cancer'])
plt.title('Final Model Confusion Matrix')
plt.xlabel('Predicted Classes')
plt.ylabel('Actual Classes')
plt.show()
```


    
![png](output_10_0.png)
    



```python
# Creation of confusion matrix for AdaBoost
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_test, y_test_pred_adaboost)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Cancer', 'Cancer'],
            yticklabels=['No Cancer', 'Cancer'])
plt.title('Final Model Confusion Matrix')
plt.xlabel('Predicted Classes')
plt.ylabel('Actual Classes')
plt.show()
```


    
![png](output_11_0.png)
    
