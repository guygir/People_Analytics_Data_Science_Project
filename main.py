# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def train(excel_file):
    # Read data from Excel into a DataFrame
    df = pd.read_excel(excel_file)
    df.drop(columns=['חותמת זמן'])
    filtered_df = df[(df['תואר'] == 'מדעי המחשב') & (df['מוסד'] == 'אוניברסיטת תל אביב')]
    filtered_df['שנה'] = pd.to_numeric(filtered_df['שנה'], errors='coerce')
    filtered_df['נשארו'] = pd.to_numeric(filtered_df['נשארו'], errors='coerce')
    filtered_df['duration of studies'] = filtered_df['שנה'] + ((filtered_df['נשארו'] - 1) / 2)
    filtered_df.to_excel('test1.xlsx', index=False)

    # Split data into features (X) and target variable (y)
    X = filtered_df.drop(columns=['duration of studies'])
    y = filtered_df['duration of studies']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.8)

    # Identify categorical features
    categorical_features_indices = np.where(X.dtypes != float)[0]

    # Create and train the CatBoost classifier
    model = CatBoostClassifier(logging_level='Verbose')
    train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
    fm = model.fit(train_pool, logging_level='Verbose', plot=True)

    # Get feature importances
    feature_importances = fm.get_feature_importance(train_pool)
    feature_names = X_train.columns

    # Print feature importance scores
    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
        print(f'{name}: {score}')

    # Make predictions on the test set
    y_pred = fm.predict(X_test)

    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualize confusion matrix
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Print evaluation metrics
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')


def main():
    excel_file = "resclean.xlsx"
    train(excel_file)

if __name__ == "__main__":
    main()