import pandas as pd

from loguru import logger
from sqlalchemy import select

from config import engine
from db.db_model import RentApartments


def load_data_from_db() -> pd.DataFrame:
    logger.info('Extracting csv file from database')
    query = select(RentApartments)
    return pd.read_sql(
        query,
        engine,
    )
