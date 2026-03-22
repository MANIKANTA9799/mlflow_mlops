import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd

# init dagshub (IMPORTANT)
dagshub.init(
    repo_owner='mytab123yy',
    repo_name='mlflow_mlops',
    mlflow=True
)

# load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)#type:ignore 
y = pd.Series(data.target, name='target')#type:ignore 

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
rf = RandomForestClassifier(random_state=42)

# param grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# grid search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2
)

mlflow.set_experiment('breast-cancer-rf-hp')

with mlflow.start_run(run_name="gridsearch_parent"):

    grid_search.fit(X_train, y_train)

    # log all param combinations
    for i in range(len(grid_search.cv_results_['params'])):

        params = grid_search.cv_results_["params"][i]
        mean_score = grid_search.cv_results_["mean_test_score"][i]
        std_score = grid_search.cv_results_["std_test_score"][i]
        rank = grid_search.cv_results_["rank_test_score"][i]

        with mlflow.start_run(
            nested=True,
            run_name=f"run_{i}_rank_{rank}"
        ):
            mlflow.log_params(params)
            mlflow.log_metric("mean_accuracy", mean_score)
            mlflow.log_metric("std_accuracy", std_score)
            mlflow.log_metric("rank", rank)

    # best model
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", best_score)

    # log model
    mlflow.sklearn.log_model(#type:ignore 
        grid_search.best_estimator_,
        "best_random_forest"
    )

    print(best_params)
    print(best_score)