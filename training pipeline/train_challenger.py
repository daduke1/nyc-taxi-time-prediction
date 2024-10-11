import pickle
import mlflow
import pathlib
import dagshub
import numpy as np
import pandas as pd
import mlflow.sklearn
from mlflow import MlflowClient
from hyperopt.pyll import scope
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction import DictVectorizer
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from prefect import flow, task
from sklearn.metrics import mean_squared_error

@task(name="Read Data", retries=4, retry_delay_seconds=[1, 4, 8, 16])
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task(name="Add features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  # 'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values
    return X_train, X_val, y_train, y_val, dv

@task(name="Hyperparameter Tuning Gradient Boosting")
def hyper_parameter_tuning_gb(X_train, X_val, y_train, y_val, dv):
    mlflow.sklearn.autolog()

    training_dataset = mlflow.data.from_numpy(X_train.data, targets=y_train, name="green_tripdata_2024-01")

    validation_dataset = mlflow.data.from_numpy(X_val.data, targets=y_val, name="green_tripdata_2024-02")

    # Define search space for Gradient Boosting Regressor
    gb_search_space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'random_state': 42
    }

    def objective(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "GradientBoosting")
            mlflow.log_params(params)

            # Train Gradient Boosting model
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)

            # Log the trained model
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Predictions and RMSE on validation set
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    with mlflow.start_run(run_name="GradientBoosting Hyperparameter Optimization", nested=True):
        gb_trials = Trials()
        best_gb_params = fmin(
            fn=objective,
            space=gb_search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=gb_trials
        )

        # Convert the best params back to int where necessary
        best_gb_params["n_estimators"] = int(best_gb_params["n_estimators"])
        best_gb_params["max_depth"] = int(best_gb_params["max_depth"])

        mlflow.log_params(best_gb_params)

    return best_gb_params

@task(name="Hyperparameter Tuning Random Forest")
def hyper_parameter_tuning_rf(X_train, X_val, y_train, y_val, dv):
    mlflow.sklearn.autolog()

    # Define search space for Random Forest
    rf_search_space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 1)),
        'max_depth': scope.int(hp.quniform('max_depth', 4, 20, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 5, 1)),
        'random_state': 42
    }

    # Define the objective function for Random Forest
    def objective(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "RandomForest")
            mlflow.log_params(params)

            # Train Random Forest model
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)

            # Log the trained model
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Predictions and RMSE on validation set
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    # Hyperparameter Optimization for Random Forest
    with mlflow.start_run(run_name="RandomForest Hyperparameter Optimization", nested=True):
        rf_trials = Trials()
        best_rf_params = fmin(
            fn=objective,
            space=rf_search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=rf_trials
        )

        # Convert the best params back to int where necessary
        best_rf_params["n_estimators"] = int(best_rf_params["n_estimators"])
        best_rf_params["max_depth"] = int(best_rf_params["max_depth"])
        best_rf_params["min_samples_split"] = int(best_rf_params["min_samples_split"])
        best_rf_params["min_samples_leaf"] = int(best_rf_params["min_samples_leaf"])

        mlflow.log_params(best_rf_params)

    return best_rf_params

@task(name="Train best RF model")
def train_best_rf_model(X_train, X_val, y_train, y_val, dv, best_params) -> None:
    """Train a Random Forest model with best hyperparameters and log the results."""

    with mlflow.start_run(run_name="Best Random Forest Model"):
        mlflow.log_params(best_params)

        # Train the RandomForest model
        model = RandomForestRegressor(**best_params)
        model.fit(X_train, y_train)

        # Make predictions and calculate RMSE
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mlflow.log_metric("rmse", rmse)

        # Save and log the preprocessor
        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor_rf.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        # Log artifacts to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact("models/preprocessor_rf.b", artifact_path="preprocessor")

    return None

@task(name="Train best GB model")
def train_best_gb_model(X_train, X_val, y_train, y_val, dv, best_params) -> None:
    """Train a Gradient Boosting model with best hyperparameters and log the results."""

    with mlflow.start_run(run_name="Best Gradient Boosting Model"):
        mlflow.log_params(best_params)

        # Train the GradientBoosting model
        model = GradientBoostingRegressor(**best_params)
        model.fit(X_train, y_train)

        # Make predictions and calculate RMSE
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred, squared=False))
        mlflow.log_metric("rmse", rmse)

        # Save and log the preprocessor
        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor_gb.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        # Log artifacts to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact("models/preprocessor_gb.b", artifact_path="preprocessor")

    return None

@task(name = 'Register Model')
def register_model():
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    df = mlflow.search_runs(order_by=['metrics.rmse'])
    run_id = df.loc[df['metrics.rmse'].idxmin()]['run_id']
    run_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=run_uri,
        name="nyc-taxi-model-prefect"
    )
    model_name = "nyc-taxi-model-prefect"
    model_version_alias = "challenger"

    # create "challenger" alias for version of model "nyc-taxi-model-prefect"
    client.set_registered_model_alias(
        name=model_name,
        alias=model_version_alias
    )

@flow(name="HW Flow")
def hw_flow(year: str, month_train: str, month_val: str) -> None:
    """The main training pipeline"""

    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"

    # MLflow settings
    dagshub.init(url="https://dagshub.com/daduke1/nyc-taxi-time-prediction", mlflow=True)

    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name="nyc-taxi-experiment-prefect")

    # Load
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    # Hyper-parameter RF tuning
    best_rf_params = hyper_parameter_tuning_rf(X_train, X_val, y_train, y_val, dv)

    # Train best RF model
    train_best_rf_model(X_train, X_val, y_train, y_val, best_rf_params)

    # Hyper-parameter GB tuning
    best_gb_params = hyper_parameter_tuning_gb(X_train, X_val, y_train, y_val, dv)

    # Train best GB model
    train_best_gb_model(X_train, X_val, y_train, y_val, best_gb_params)

    # Register model
    register_model()