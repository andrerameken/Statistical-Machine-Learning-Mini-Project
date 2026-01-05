import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.model_selection as skl_ms
import sklearn.metrics as skl_mtr


df = pd.read_csv("train.csv").dropna()

# Imbalance in the data.
# print(df["Lead"].value_counts())
# Male: 785
# Female: 254
# Choose female as positive class.

df["Lead"].replace(["Female", "Male"], [1, 0], inplace=True)

X = df.drop("Lead", axis=1)
Y = df["Lead"]
# Split data
X_train, X_test, y_train, y_test = skl_ms.train_test_split(X, Y, test_size=0.25, random_state = 42)

# Scale the data
scaler = skl_pre.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = skl_lm.LogisticRegression()

# Since there is imbalance in data, compare no class weight with balanced. Perhaps better result?
param_grid = {'C': np.logspace(-5, 5, 30), 
        'class_weight' : [None, "balanced"]}

# Do gridsearch for optimal regularization and class_weight.

clf = skl_ms.GridSearchCV(model, param_grid, scoring ="accuracy", cv = 10)  # Maybe use f1 as scoring?
clf.fit(X_train_scaled, y_train)
optimal_model = clf.best_estimator_
print(clf.best_estimator_)
optimal_model.fit(X_train_scaled, y_train)
#print(clf1.best_estimator_)
#print(clf1.best_params_)
#print(clf1.best_score_)

test_prediction = optimal_model.predict(X_test_scaled)
print(f"ROC-AUC score: {skl_mtr.roc_auc_score(y_test, test_prediction)}") # Should maybe use area under precision-recall curve
print(f"Accuracy on test data: {skl_mtr.accuracy_score(y_test, test_prediction)}")
print(f"F1 score on test data: {skl_mtr.f1_score(y_test, test_prediction)} ")
print(skl_mtr.confusion_matrix(y_test, test_prediction))

test = pd.read_csv("test.csv").dropna()
X_real_test = scaler.transform(test)
y_pred = optimal_model.predict(X_real_test)
df2 = pd.DataFrame(y_pred).T
df2.to_csv("predictions.csv", index=False, chunksize=1)