import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import optuna
import joblib

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

# Optuna's objective function
def objective(trial):
    # Print the start of the trial
    print(f"Starting Optuna trial with trial number: {trial.number}")

    # Hyperparameters to tune
    max_depth = trial.suggest_int('max_depth', 3, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)

    # Print the parameters for the current trial
    print(f"Trial {trial.number}: max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")

    # Start a new nested run for each trial
    with mlflow.start_run(nested=True):
        # Train Decision Tree model with tuned hyperparameters
        model = DecisionTreeClassifier(max_depth=max_depth,
                                       min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters and metrics to MLflow
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_metric("accuracy", accuracy)

        # Print trial result
        print(f"Trial {trial.number} completed with accuracy: {accuracy:.4f}")
    return accuracy

# Run Optuna optimization
print("Starting Optuna optimization...")
mlflow.start_run()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Log the best hyperparameters
best_params = study.best_params
print(f"Best hyperparameters found: {best_params}")
mlflow.log_param("best_max_depth", best_params["max_depth"])
mlflow.log_param("best_min_samples_split", best_params["min_samples_split"])
mlflow.log_param("best_min_samples_leaf", best_params["min_samples_leaf"])

# Log the dataset as an artifact
mlflow.log_artifact("Data/DecisionTree_dataset.csv")

# Log the best model
best_model = DecisionTreeClassifier(
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"]
)
best_model.fit(X_train, y_train)
mlflow.sklearn.log_model(best_model, "decision_tree_model")

# Save the best model to a file
joblib.dump(best_model, "best_decision_tree_model.pkl")

# End the MLflow run
mlflow.end_run()

print("Training complete. Best Hyperparameters:", best_params)
