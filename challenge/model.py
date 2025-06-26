import logging
import joblib
import pandas as pd
import numpy as np

from typing import Tuple, Union, List
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from utils.path import RAW_DATA_DIR, MODEL_PATH

class DelayModel:

    def __init__(
        self
    ):
        self._model = self._load_model(MODEL_PATH) # Model should be saved in this attribute.
        self.top_features = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]
    
    def get_min_diff(self, data):
        """
        Calculate the difference in minutes between two datetime columns in a
        DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            A DataFrame containing two columns:
            - 'Fecha-O': Date and time of flight operation, a datetime column or string in the
            format '%Y-%m-%d %H:%M:%S'
            - 'Fecha-I': Scheduled date and time of the flight, a datetime column or string in the
            format '%Y-%m-%d %H:%M:%S'

        Returns
        -------
        pd.Series
            A Series containing the difference in minutes between 'Fecha-O'
        and 'Fecha-I' for each row.
        """
        
        fecha_o = pd.to_datetime(data['Fecha-O'])
        fecha_i = pd.to_datetime(data['Fecha-I'])

        return (fecha_o - fecha_i).dt.total_seconds() / 60
    
    def _load_model(self, model_path: Path):
        """
        Load Logistic Regression Model Saved

        Parameters
        model_path: Path to the saved model
        """

        logging.info(f"Loading Model from {model_path}")

        try:
            model = joblib.load(model_path)
            logging.info("Model loaded")

            return model

        except Exception as e:
            logging.warning(f"No model found in {model_path}: {e}")
    
    def _save_model(self, model, model_path) -> None:
        # Make sure folder exists
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        data = data.copy()
        
        if target_column:
            # Only compute target if in training mode
            data['min_diff']  = self.get_min_diff(data)
            threshold_in_minutes = 15
            data[target_column] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

        # Encode categorical variables
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
            )
        
        # Asegurar que estén todas las columnas esperadas
        features = features.reindex(columns=self.top_features, fill_value=0)
        
        if target_column:
            target = data[[target_column]]
            return features, target

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        target_column = target.columns[0]

        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)
        n_y0 = len(y_train[y_train[target_column] == 0])
        n_y1 = len(y_train[y_train[target_column] == 1])
        scale_pos_weight = n_y0/n_y1
        logging.info(f"Scale: {scale_pos_weight}")

        model = LogisticRegression(class_weight={1: n_y0/len(y_train), 0: n_y1/len(y_train)})
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)

        logging.info(confusion_matrix(y_test, y_pred))
        logging.info(classification_report(y_test, y_pred))

        # Save the model
        self._save_model(model, MODEL_PATH)
        self._model = model

        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            logging.warning("No model loaded, loading now...")
            self._load_model(MODEL_PATH)
            if self._model is None:
                raise ValueError("No model available for prediction.")
        
        return self._model.predict(features).tolist()