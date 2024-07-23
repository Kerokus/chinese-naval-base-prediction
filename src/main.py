"""
This script will import the model_data_updated.csv,
clean up the columns, train the model, and export the model/scaler.
These pickel files will be used for a Streamlit frontend later.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


def get_clean_data():
    data = pd.read_csv('data/model_data_updated.csv')
    data = data.drop(["port_name", "Recipient", "region", "latitude",
                     "longitude", "received_chinese_investment"], axis=1)
    return data


def create_and_train_model(data):
    y = data['potential_chinese_naval_use']
    X = data.drop(['potential_chinese_naval_use'], axis=1)

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=42
    )

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print('Model Accuracy: ', accuracy_score(y_test, y_pred))
    print('Classification Report: \n', classification_report(y_test, y_pred))

    return model, scaler


def main():
    data = get_clean_data()
    model, scaler = create_and_train_model(data)

    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/scalers.pkl', 'wb') as f:
        pickle.dump(scaler, f)


if __name__ == '__main__':
    main()
