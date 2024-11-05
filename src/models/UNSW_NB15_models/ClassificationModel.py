from src.data.UNSW_NB15_preprocessor.Preprocessor import Preprocessor
import pandas as pd


class ClassificationModel:
    """
    CatBoost model for predicting the attack category.

    Attributes:
        data_path (str): The path to the input data.
        model_path (str): The path to the trained model.
        preprocessor (Preprocessor): The preprocessor object.
        model (CatBoostClassifier): The trained CatBoost model.

    Methods:
        predict: Make predictions using the trained CatBoost model.

    """
    def __init__(self):
        """
        Initialize the CatModel with the path to the input data.

        Args:
            None

        Returns:
            None
        """
        self.model_path = "models/UNSW_NB15_models/classification_model_84_F1_V2.cbm"
        self.preprocessor = Preprocessor(self.model_path)
        self.model = self.preprocessor.model

    def predict(self, df : pd.DataFrame):
        """
        Make predictions using the trained CatBoost model.

        Args:
            data_path (str): The path to the input data.

        Returns:
            predictions (np.array): The predicted attack categories.

        """
        # Preprocess the input data
        df = self.preprocessor.preprocess(df)
        df = self.preprocessor.create_pool(df)

        # Make predictions
        predictions = self.model.predict(df)
        return predictions
