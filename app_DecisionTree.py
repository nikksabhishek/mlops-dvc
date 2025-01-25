import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Set the tracking URI to the MLflow server
mlflow.set_tracking_uri('http://localhost:5000')

# Load data
data = load_iris()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Save the dataset locally (optional: for artifact logging)
X_with_target = pd.concat([X, y], axis=1)
X_with_target.to_csv("Data/DecisionTree_dataset.csv", index=False)

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

# Experiment tracking with MLflow
mlflow.set_experiment("Decision Tree - Iris Classification Experiment")

mlflow.start_run()

# Train Decision Tree model
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Log parameters and metrics
mlflow.log_param("model_type", "Decision Tree")
mlflow.log_param("max_depth", 5)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log parameters, metrics, and model
mlflow.log_metric("accuracy", accuracy)

# Log the dataset as an artifact
mlflow.log_artifact("Data\DecisionTree_dataset.csv")

# Log model
mlflow.sklearn.log_model(model, "decision_tree_model")

mlflow.end_run()
