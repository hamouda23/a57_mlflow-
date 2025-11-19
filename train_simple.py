import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)
model = RandomForestRegressor(n_estimators=10, random_state=42)

with mlflow.start_run():
    mlflow.log_param("n_estimators", 10)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
    print("mse", mse)
