
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Dummy SSA function to simulate behavior for testing
def SSA(objf, lb, ub, dim, N, Max_iteration):
    """
    Simulated SSA function for testing.
    Finds a random solution within the bounds.
    """
    best_solution = np.random.uniform(lb, ub)
    best_fitness = objf(best_solution)
    return type('solution', (object,), {
        "bestIndividual": best_solution,
        "bestFitness": best_fitness
    })()

# Load dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
X = breast_cancer_dataset.data
y = breast_cancer_dataset.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function
def objective_function(params):
    """
    Objective function for SSA. It takes hyperparameters as input and returns the error rate.
    params: list of hyperparameters [C, max_iter]
    """
    C, max_iter = params
    try:
        model = LogisticRegression(C=C, max_iter=int(max_iter), solver='lbfgs', random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return 1 - accuracy  # Minimize the error rate
    except Exception as e:
        return float('inf')  # Return a large error if the model fails

# Define bounds for hyperparameters
lb = [0.01, 50]  # Lower bounds for C and max_iter
ub = [100, 500]  # Upper bounds for C and max_iter

# Run SSA to optimize hyperparameters
dim = 2  # Number of hyperparameters
num_agents = 30  # Number of agents
max_iterations = 50  # Maximum number of iterations

# Run SSA
best_solution = SSA(objective_function, lb, ub, dim, num_agents, max_iterations)

# Train the Logistic Regression model with optimized hyperparameters
best_C, best_max_iter = best_solution.bestIndividual
optimized_model = LogisticRegression(C=best_C, max_iter=int(best_max_iter), solver='lbfgs', random_state=42)
optimized_model.fit(X_train, y_train)

# Evaluate the optimized model
optimized_predictions = optimized_model.predict(X_test)
optimized_accuracy = accuracy_score(y_test, optimized_predictions)

# Results
print(f"Optimized Hyperparameters: C: {best_C}, Max Iterations: {int(best_max_iter)}")
print(f"Optimized Accuracy: {optimized_accuracy}")
