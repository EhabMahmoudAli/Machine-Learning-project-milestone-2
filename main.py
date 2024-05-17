import pandas as pd
# import seaborn as sns
import numpy as np
# import matplotlib.pyplot as plt
# import sklearn
# from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import VotingClassifier  # RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# loading data
data = pd.read_csv('ElecDeviceRatingPrediction_Milestone2.csv')
label_encoder_y = LabelEncoder()
data['rating'] = label_encoder_y.fit_transform(data['rating'])
X = data.drop(columns=['rating'])
Y = data['rating']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=50)
null_counts = data.isnull().sum()
# print(null_counts)
# print(X_test.columns)
# nulls values
X_train_numeric = X_train.select_dtypes(include=np.number)
X_train_non_numeric = X_train.select_dtypes(exclude=np.number)
# print( "X_train_numeric)",X_train_numeric.columns)
# print("X_train_non_numeric)",X_train_non_numeric.columns)
median_values = X_train_numeric.median()
X_train[X_train_numeric.columns] = X_train_numeric.fillna(median_values)
mode_values = X_train_non_numeric.mode().iloc[0]
X_train[X_train_non_numeric.columns] = X_train_non_numeric.fillna(mode_values)

# encoding

# Perform one hot encoding
oneHotEncoded = pd.get_dummies(X_train['processor_brand'], prefix='processor_brand')
encodedInt = oneHotEncoded.astype(int)
X_train = pd.concat([X_train, encodedInt], axis=1)
X_train.drop('processor_brand', axis=1, inplace=True)

oneHotEncode_test = pd.get_dummies(X_test['processor_brand'], prefix='processor_brand')
encodedInt = oneHotEncode_test.astype(int)
X_test = pd.concat([X_test, encodedInt], axis=1)
X_test.drop('processor_brand', axis=1, inplace=True)

encodedColumns = ['brand', 'processor_name', 'processor_gnrtn', 'ram_gb', 'ram_type',
                  'ssd', 'hdd', 'os', 'graphic_card_gb', 'weight', 'warranty',
                  'Touchscreen', 'msoffice']
label_encoder = LabelEncoder()

for column in encodedColumns:
    X_train[column] = label_encoder.fit_transform(X_train[column])
    X_test[column] = label_encoder.transform(X_test[column])


# print(Y_train['rating']
# print(X_train['processor_name'])
# sns.boxplot(X_test,palette="rainbow",orient='h')
# plt.show()
def remove_outliers(X):
    numerical_columns = X.select_dtypes(include=[np.number]).columns

    for column in numerical_columns:
        if X[column].nunique() > 2:  # for non-binary data / categorical
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            lowerlimit = Q1 - (1.5 * IQR)
            upperlimit = Q3 + (1.5 * IQR)
            X.loc[(X[column] < lowerlimit) | (X[column] > upperlimit), column] = np.nan
    X[numerical_columns] = X[numerical_columns].fillna(X[numerical_columns].median())
    return X


X_train = remove_outliers(X_train)
X_test = remove_outliers(X_test)
# sns.boxplot(X_test,palette="rainbow",orient='h')
# plt.show()
# print(X_train['processor_brand_intell'].values)


# Scaling features using Min-Max Scaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# feature selection wrapper
estimator = LogisticRegression()
rfe = RFE(estimator, n_features_to_select=6)
rfe.fit(X_train_scaled, Y_train)
selected_features = X_train.columns[rfe.support_]
# print("Selected Features:")
# print(selected_features)

# print(X_test.columns)
print('//////////////////////////')
print('\n------------------ logistic regression --------------------')
#
model = LogisticRegression(C=3)  # best c=3
model.fit(X_train_scaled, Y_train)

y_pred_train = model.predict(X_train_scaled)
train_accuracy = accuracy_score(Y_train, y_pred_train)
print(f"Logistic Regression Training Accuracy: {train_accuracy}")

y_pred_test = model.predict(X_test_scaled)
test_accuracy = accuracy_score(Y_test, y_pred_test)
print(f"Logistic Regression Test Accuracy: {test_accuracy}")

print('\n------------------ svm --------------------')
# , 'rbf', 'poly
kernels = ['linear']  # best one is poly (hyperparameter)
for kernel in kernels:
    svc = SVC(kernel=kernel, C=6)  # best at c=6
    svc.fit(X_train_scaled, Y_train)
    y_train_pred = svc.predict(X_train_scaled)
    print(f"\nEvaluation Metrics for Training Set with {kernel} kernel:")
    print("Accuracy:", accuracy_score(Y_train, y_train_pred))
    print("F1-score:", f1_score(Y_train, y_train_pred, average='weighted', zero_division=1))
    y_test_pred = svc.predict(X_test_scaled)
    print(f"\nEvaluation Metrics for Test Set with {kernel} kernel:")
    print("Accuracy:", accuracy_score(Y_test, y_test_pred))
    print("F1-score:", f1_score(Y_test, y_test_pred, average='weighted', zero_division=1))

print("\n----------------- STACKING ---------------------")
# Define classifiers
classifiers = [
    ('LogisticRegression', LogisticRegression()),
    # ('DecisionTree', DecisionTreeClassifier()),
    # ('Naive Bayes', GaussianNB()),
    ('KNeighbours', KNeighborsClassifier()),
    ('SVM', SVC()),
]

# Create the stacked classifier
stacked_clf = StackingClassifier(
    estimators=classifiers,
    final_estimator=LogisticRegression()
)
stacked_clf.fit(X_train_scaled, Y_train)
train_predictions = stacked_clf.predict(X_train_scaled)

train_accuracy = accuracy_score(Y_train, train_predictions)
print("Stacked Classifier Train Accuracy:", train_accuracy)

test_predictions = stacked_clf.predict(X_test_scaled)

test_accuracy = accuracy_score(Y_test, test_predictions)
print("Stacked Classifier Test Accuracy:", test_accuracy)

print("\n----------------VOTING---------------------")

# Define the base classifiers for the voting approach
classifiers = [
    ('tree', DecisionTreeClassifier()),
    ('LOG', LogisticRegression()),
    # ('Naive Bayes', GaussianNB()),
    # ('KNeighbours', KNeighborsClassifier()),
    ('svm', SVC())
]

voting_clf = VotingClassifier(estimators=classifiers, voting='hard')
voting_clf.fit(X_train_scaled, Y_train)
y_pred = voting_clf.predict(X_test_scaled)

train_accuracy = accuracy_score(Y_train, voting_clf.predict(X_train_scaled))
print("Voting Classifier Training Accuracy:", train_accuracy)

test_accuracy = accuracy_score(Y_test, y_pred)
print("Voting Classifier Test Accuracy:", test_accuracy)
print('\n------------------ KNN  --------------------')
k = 21  # best 21
knn_clf = KNeighborsClassifier(
    n_neighbors=k)  # in document explain 3 choices for hyperparameter by comparing test & train acc for each choice
knn_clf.fit(X_train_scaled, Y_train)

train_predictions = knn_clf.predict(X_train_scaled)
train_accuracy = accuracy_score(Y_train, train_predictions)
print("KNN Training Accuracy:", train_accuracy)

test_predictions = knn_clf.predict(X_test_scaled)
test_accuracy = accuracy_score(Y_test, test_predictions)
print("KNN Test Accuracy:", test_accuracy)
