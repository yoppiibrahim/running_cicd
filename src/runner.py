from loguru import logger

from model.model_service import ModelService


@logger.catch
def main():
    svc_model = ModelService()
    svc_model.load_model()
    feature_values = {
        'area': 85,
        'constraction_year': 2015,
        'bedrooms': 2,
        'garden': 20,
        'balcony_yes': 1,
        'parking_yes': 1,
        'furnished_yes': 0,
        'garage_yes': 0,
        'storage_yes': 1,

    }
    pred = svc_model.predict(list(feature_values.values()))
    logger.info(f'Prediction :{pred}')


if __name__ == '__main__':
    main()
