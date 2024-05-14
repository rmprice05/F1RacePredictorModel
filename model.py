import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def fetch_season_data(start_year, end_year):
    """Fetches race results for each race participant."""
    all_results = []
    for year in range(start_year, end_year + 1):
        url = f"http://ergast.com/api/f1/{year}/results.json?limit=1000"
        response = requests.get(url)
        data = response.json()
        races = data['MRData']['RaceTable']['Races']
        for race in races:
            race_name = race['raceName']
            date = race['date']
            for result in race['Results']:
                all_results.append({
                    'race_name': race_name,
                    'date': date,
                    'driver_id': result['Driver']['driverId'],
                    'driver': result['Driver']['familyName'],
                    'constructor': result['Constructor']['name'],
                    'grid_position': int(result['grid']),
                    'finish_position': int(result['positionText'].replace('R', '').replace('D', '') if result['positionText'].isdigit() else 0),
                    'points': float(result['points']),
                    'laps': int(result['laps']),
                    'status': result['status'],
                    'win': result['position'] == '1'
                })
    return pd.DataFrame(all_results)

def fetch_qualifying_results(start_year, end_year):
    """Fetches qualifying results for each race participant."""
    all_qualifying_results = []
    for year in range(start_year, end_year + 1):
        url = f"http://ergast.com/api/f1/{year}/qualifying.json?limit=1000"
        response = requests.get(url)
        data = response.json()
        races = data['MRData']['RaceTable']['Races']
        for race in races:
            date = race['date']
            for result in race['QualifyingResults']:
                all_qualifying_results.append({
                    'date': date,
                    'driver_id': result['Driver']['driverId'],
                    'qualifying_position': int(result['position']),
                })
    return pd.DataFrame(all_qualifying_results)

def preprocess_data(f1_data, qualifying_data):
    """Integrates qualifying results and processes data for modeling."""
    f1_data['date'] = pd.to_datetime(f1_data['date'])
    qualifying_data['date'] = pd.to_datetime(qualifying_data['date'])
    full_data = pd.merge(f1_data, qualifying_data, on=['date', 'driver_id'], how='left')
    full_data = pd.get_dummies(full_data, columns=['driver', 'constructor', 'status'])
    return full_data

def main():
    start_year = 1980
    current_year = pd.to_datetime('now').year - 1
    f1_race_data = fetch_season_data(start_year, current_year)
    f1_qualifying_data = fetch_qualifying_results(start_year, current_year)
    f1_data = preprocess_data(f1_race_data, f1_qualifying_data)

    # Further feature engineering if needed

    X = f1_data.drop(['win', 'race_name', 'date', 'driver_id'], axis=1)
    y = f1_data['win'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
