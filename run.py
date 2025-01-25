import subprocess
import mlflow

# Set the tracking URI to the MLflow server
mlflow.set_tracking_uri('http://localhost:5000')

# Number of times to run the scripts
num_runs = 1

# Paths to your scripts
decision_tree_script = "app_DecisionTree.py"
svm_script = "app_SVM.py"
gradient_Boost_script = "app_GradientBoost.py"
knn_script = "app_KNN.py"
logistic_regression_script = "app_LogisticRegression.py"

for i in range(num_runs):
    print(f"Running Experiment {i+1}/{num_runs}...")
    
    # Run the Decision Tree script
    subprocess.run(["python", decision_tree_script], check=True)
    
    # Run the SVM script
    subprocess.run(["python", svm_script], check=True)
    
    # Run the Gradient Boosting script
    subprocess.run(["python", gradient_Boost_script], check=True)
    
    # Run the KNN script
    subprocess.run(["python", knn_script], check=True)
    
    # Run the Logistic Regression script
    subprocess.run(["python", logistic_regression_script], check=True)

print(f"Completed {num_runs} runs.")
