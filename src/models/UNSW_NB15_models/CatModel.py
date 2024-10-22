from src.data.UNSW_NB15_preprocessor.Preprocessor import ModelPreprocessor
class CatModel:
    """
    CatBoost model for predicting the attack category.

    Attributes:
        data_path (str): The path to the input data.
        model_path (str): The path to the trained model.
        preprocessor (ModelPreprocessor): The preprocessor object.
        model (CatBoostClassifier): The trained CatBoost model.

    Methods:
        predict: Make predictions using

    """
    def __init__(self, data_path):
        """
        Initialize the CatModel with the path to the input data.

        Args:
            data_path (str): The path to the input data.

        Returns:
            None
        """
        self.data_path = data_path
        self.model_path = "../../models/UNSW_NB15_models/catboost_model_94.5_Recall.cbm"
        self.preprocessor = ModelPreprocessor(self.data_path, self.model_path)
        self.model = self.preprocessor.model

    def predict(self):
        """
        Make predictions using the trained CatBoost model.

        Args:
            None

        Returns:
            predictions (np.array): The predicted attack categories.

        """
        # Preprocess the input data
        df = self.preprocessor.preprocess()
        df = self.preprocessor.create_pool(df)

        # Make predictions
        predictions = self.model.predict(df)
        return predictions