import pandas as pd
import pickle as pk

from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from config import model_settings
from model.pipeline.preparation import prepare_data


def build_model() -> None:
    logger.info('Starting build model Pipeline')
    df = prepare_data()

    feature_names = [
        'area',
        'constraction_year',
        'bedrooms',
        'garden',
        'balcony_yes',
        'parking_yes',
        'furnished_yes',
        'garage_yes',
        'storage_yes'
    ]
    X, y = _get_x_y(
        df,
        col_x=feature_names,
    )

    X_train, X_test, y_train, y_test = _split_train_test(
        X,
        y,
    )

    rf = _train_model(
        X_train,
        y_train,
    )

    _evaluation_model(
        rf,
        X_test,
        y_test,
    )

    _save_model(rf)


def _get_x_y(
        dataframe: pd.DataFrame,
        col_x: pd.Series,
        col_y: str = 'rent',
) -> tuple[pd.DataFrame, pd.Series]:

    logger.info(f'Defining X and y Variable, X var{col_x} y Var {col_y}')
    return dataframe[col_x], dataframe[col_y]


def _split_train_test(
        features: pd.DataFrame,
        target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    logger.info('Spliting data into train and test')
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2, # noqa:  WPS432
    )

    return X_train, X_test, y_train, y_test


def _train_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
) -> RandomForestRegressor:

    logger.info('Training model using Hyperparameter tuning')
    grid_space = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9, 12]
    }

    logger.debug(f'Grid Space: {grid_space}')
    grid = GridSearchCV(
        RandomForestRegressor(),
        param_grid=grid_space,
        cv=5,
        scoring='r2',
    )

    model_grid = grid.fit(
        X_train,
        y_train,
    )

    return model_grid.best_estimator_


def _evaluation_model(
        model: RandomForestRegressor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
) -> float:

    model_score = model.score(
        X_test,
        y_test,
    )

    logger.info(f'Evaluation performance Score: {model_score}')
    return model_score


def _save_model(model: RandomForestRegressor) -> None:
    model_path = f'{model_settings.model_path}/{model_settings.model_name}'
    logger.info(f'Saving a model at Directory {model_path}')

    with open(model_path, 'wb') as model_file:
        pk.dump(model, model_file)
