import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess data
df = pd.read_csv('/Users/naniankala/Desktop/PS_20174392719_1491204439457_log.csv')
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig',
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})
df = df.drop(['isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1)

# Label encoding for 'type' column
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])

# Feature engineering
df['errorBalanceOrig'] = df['newBalanceOrig'] + df['amount'] - df['oldBalanceOrig']
df['errorBalanceDest'] = df['oldBalanceDest'] + df['amount'] - df['newBalanceDest']

# Split data into features (X) and target (Y)
Y = df['isFraud']
X = df.drop('isFraud', axis=1)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Standardizing features for KNN
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# K-Nearest Neighbors Model
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
print("*********** K-Nearest Neighbors **********")
knn_report = classification_report(Y_test, Y_pred_knn, output_dict=True)
print("Precision:", knn_report['1']['precision'])
print("Recall:", knn_report['1']['recall'])
print("F-Score:", knn_report['1']['f1-score'])
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_knn))

# Random Forest Model
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1, n_jobs=-1, class_weight="balanced")
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
print("\n*********** Random Forest **********")
rf_report = classification_report(Y_test, Y_pred_rf, output_dict=True)
print("Precision:", rf_report['1']['precision'])
print("Recall:", rf_report['1']['recall'])
print("F-Score:", rf_report['1']['f1-score'])
print("Confusion Matrix:\n", confusion_matrix(Y_test, Y_pred_rf))