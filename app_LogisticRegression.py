import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pandas as pd

# Set the tracking URI to the MLflow server
mlflow.set_tracking_uri('http://localhost:5000')

# Load Iris dataset
data = load_iris()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Save the dataset locally (optional: for artifact logging)
X_with_target = pd.concat([X, y], axis=1)
X_with_target.to_csv("Data/LogisticRegression_dataset.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Experiment tracking with MLflow
mlflow.set_experiment("Logistic Regression - Iris Classification Experiment")

# Experiment tracking with MLflow
mlflow.start_run()

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Log parameters and metrics
mlflow.log_param("model_type", "Logistic Regression")
mlflow.log_param("max_iter", 200)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log parameters, metrics, and model
mlflow.log_metric("accuracy", accuracy)

# Log the dataset as an artifact
mlflow.log_artifact("Data\LogisticRegression_dataset.csv")

# Log model
mlflow.sklearn.log_model(model, "logistic_regression_model")

mlflow.end_run()
