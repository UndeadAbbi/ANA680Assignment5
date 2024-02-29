import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(red_wine_path, white_wine_path):
    red_wine = pd.read_csv(red_wine_path, sep=';')
    white_wine = pd.read_csv(white_wine_path, sep=';')
    
    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'
    
    combined_data = pd.concat([red_wine, white_wine], axis=0)
    
    return combined_data

def preprocess_data(data):
    data['wine_type'] = data['wine_type'].map({'red': 0, 'white': 1})
    
    X = data.drop('quality', axis=1)
    y = data['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

def save_model_scaler(model, scaler, model_path='random_forest_model.joblib', scaler_path='scaler.joblib'):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def main():
    red_wine_path = 'dataset/winequality-red.csv'  # Update this path
    white_wine_path = 'dataset/winequality-white.csv'  # Update this path

    data = load_data(red_wine_path, white_wine_path)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(data)
    
    model = train_model(X_train_scaled, y_train)
    
    evaluate_model(model, X_test_scaled, y_test)
    
    save_model_scaler(model, scaler)

if __name__ == '__main__':
    main()