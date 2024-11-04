import pandas as pd
import numpy as np
import src.features.UNSW_NB15_features.feature_engineering as fe
from catboost import CatBoostClassifier, Pool


class ModelPreprocessor:
    """
    Preprocessor class for the CatBoost model.

    Attributes:
        df (pd.DataFrame): The input dataframe.
        categorical_features (list): List of categorical features.
        numerical_features (list): List of numerical features.
        model (CatBoostClassifier): The trained CatBoost model.
        top_prop_categories (list): List of top 'proto' categories.
        top_service_categories (list): List of top 'service' categories.
        top_state_categories (list): List of top 'state' categories.
        model_path (str): Path to the saved model.

    Methods:
        load_model_and_set_feature_names: Load the CatBoost model and retrieve the feature names.
        feature_selection: Select only the features that were used during training.
        convert_data_types: Convert the data types of the columns in the dataframe.
        selecting_categories: Select the top categories for the 'proto', 'service', and 'state' columns.
        transform_categories: Transform the categories in the 'proto', 'service', and 'state' columns.
        create_log1p_features: Create log1p features for the selected columns.
        preprocess: Preprocess the input dataframe.
        create_pool: Create a CatBoost Pool object from the input data.

    """

    def __init__(self, model_path=None):
        """
        Initialize the preprocessor with the path to the dataset and the model.

        Args:
            model_path (str): Path to the saved model.
        
        Returns:
            None

        """
        self.categorical_features = []
        self.numerical_features = []
        self.model = None
        self.top_prop_categories = None
        self.top_service_categories = None
        self.top_state_categories = None
        self.model_path = model_path  # Path to the saved model

        # If a model path is provided, load the model and retrieve feature names
        if self.model_path:
            self.load_model_and_set_feature_names()

        # Select top categories
        self.selecting_categories()

    def load_model_and_set_feature_names(self):
        """
        Load the CatBoost model and retrieve the feature names used during training.
        
        Args:
            None

        Returns:
            None

        """
        self.model = CatBoostClassifier()
        self.model.load_model(self.model_path)
        self.selected_features = self.model.feature_names_  # Get feature names from the model

    def feature_selection(self, df):
        """
        Select only the features that were used during training.

        Args:
            df (pd.DataFrame): The input dataframe.
        
        Returns:
            df (pd.DataFrame): The dataframe containing only the selected features.
        
        """
        df = df[self.selected_features] if hasattr(self, 'selected_features') else df
        return df

    def convert_data_types(self, df):
        """
        Convert the data types of the columns in the dataframe.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            df (pd.DataFrame): The dataframe with the converted data types.

        """
        self.categorical_features = ['proto', 'service', 'state']
        self.numerical_features = [col for col in df.columns if col not in self.categorical_features]
        for column in self.categorical_features:
            df[column] = df[column].astype('category')
        for column in self.numerical_features:
            df[column] = df[column].astype('float32')
        return df

    def selecting_categories(self):
        """
        Select the top categories for the 'proto', 'service', and 'state' columns.

        Args:
            None

        Returns:
            None
        """
        self.top_prop_categories = fe.top_prop_categories
        self.top_service_categories = fe.top_service_categories
        self.top_state_categories = fe.top_state_categories

    def transform_categories(self, df):
        """
        Transform the categories in the 'proto', 'service', and 'state' columns.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            df (pd.DataFrame): The dataframe with transformed categories.

        """
        df['proto'] = np.where(df['proto'].isin(self.top_prop_categories), df['proto'], '-')
        df['service'] = np.where(df['service'].isin(self.top_service_categories), df['service'], '-')
        df['state'] = np.where(df['state'].isin(self.top_state_categories), df['state'], '-')
        return df

    def create_log1p_features(self, df):
        """
        Create log1p features for the selected columns.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            df (pd.DataFrame): The dataframe with log1p features.

        """
        log_features = fe.log_features
        for feature in log_features:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature])
        return df

    def preprocess(self, data_path):
        """
        Preprocess the input dataframe.

        Args:
            data_path (str): The path to the input data.
        Returns:
            df (pd.DataFrame): The preprocessed dataframe.
            
        """
        # Load the data
        df =  pd.read_parquet(data_path)

        # Select only the features that were used during training
        df = self.feature_selection(df)

        # Transform categories
        df = self.transform_categories(df)

        # Log1p Feature Creation
        df = self.create_log1p_features(df)

        # Convert data types
        df = self.convert_data_types(df)

        return df

    def create_pool(self, X):

        """
        Create a CatBoost Pool object from the input data.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            pool (Pool): The CatBoost Pool object.

        """
        return Pool(
            data=X,
            cat_features=self.categorical_features
        )
