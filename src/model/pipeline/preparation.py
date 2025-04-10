import re

import pandas as pd
from loguru import logger

from model.pipeline.collection import load_data_from_db


def prepare_data() -> pd.DataFrame:
    logger.info('Starting data preprocessing pipeline')
    dataframe = load_data_from_db()
    data_encode = _encode_cat_cols(dataframe)
    df = _parse_garden_col(data_encode)
    return df


def _encode_cat_cols(dataframe):
    cols = ['balcony', 'parking', 'furnished', 'garage', 'storage']
    logger.info(f'Encoding categorical columns {cols}')
    return pd.get_dummies(
        dataframe,
        columns=cols,
        dtype=int,
        drop_first=True,
    )


def _parse_garden_col(dataframe):
    logger.info('Parse garden column')
    dataframe['garden'] = dataframe['garden'].apply(
        lambda x: 0 if x == 'Not present' else int(re.findall(r'/d+', x)[0])
    )
    return dataframe
