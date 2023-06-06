import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class SalesForecaster:
    def __init__(self):
        self.model = RandomForestRegressor()

    def load_data(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def preprocess_data(self, data):
        # Perform any necessary data preprocessing steps
        # such as handling missing values or encoding categorical variables
        # ...

        return data

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        return mse

    def forecast(self, X):
        predictions = self.model.predict(X)
        return predictions

# Example usage:
file_path = 'path/to/your/data.csv'

forecaster = SalesForecaster()

data = forecaster.load_data(file_path)
preprocessed_data = forecaster.preprocess_data(data)

X = preprocessed_data.drop('sales', axis=1)
y = preprocessed_data['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

forecaster.train(X_train, y_train)

mse = forecaster.evaluate(X_test, y_test)
print("Mean Squared Error:", mse)

new_data = pd.DataFrame([[5, 10, 15]], columns=['feature1', 'feature2', 'feature3'])
forecasted_sales = forecaster.forecast(new_data)
print("Forecasted Sales:", forecasted_sales)
