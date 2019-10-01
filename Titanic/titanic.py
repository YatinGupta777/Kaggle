import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


# Importing the dataset
features = ['PassengerId','Pclass','Sex',"Age","SibSp","Parch","Fare","Embarked"]
training_dataset = pd.read_csv('train.csv')

#Missing Values
# fill missing values with mean column values
training_dataset.fillna(training_dataset.mean(), inplace=True)

X_train = training_dataset[features]
y_train = training_dataset.Survived


# =============================================================================
# my_imputer = SimpleImputer()
# X_train = my_imputer.fit_transform(X_train)
# =============================================================================

test_dataset = pd.read_csv('test.csv')

# fill missing values with mean column values
test_dataset.fillna(test_dataset.mean(), inplace=True)

X_test = test_dataset[features]

one_hot_encoded_training_predictors = pd.get_dummies(X_train)
one_hot_encoded_test_predictors = pd.get_dummies(X_test)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)
# Fitting classifier to the Training set
# =============================================================================
# from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy',random_state=0)
# classifier.fit(final_train,y_train)
# =============================================================================
my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(final_train, y_train, verbose=False)


test_preds = my_model.predict(final_test)
output = pd.DataFrame({'PassengerId': final_test.PassengerId,
                       'Survived': test_preds})
output.to_csv('submission.csv', index=False)
