import pandas as pd
import numpy as np
import src.features.UNSW_NB15_features.feature_engineering as fe
from catboost import CatBoostClassifier, Pool
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class Preprocessor:
    """
    Preprocessor class for the CatBoost model.

    Attributes:
        categorical_features (list): The list of categorical features.
        numerical_features (list): The list of numerical features.
        model (CatBoostClassifier): The trained CatBoost model.
        top_prop_categories (list): The top protocol categories.
        top_service_categories (list): The top service categories.
        top_state_categories (list): The top state categories.
        model_path (str): The path to the trained model.

    Methods:
        load_model_and_set_feature_names: Load the trained model and set the feature names.
        feature_selection: Perform feature selection on the input dataframe.
        convert_data_types: Convert the data types of the columns in the input dataframe.
        selecting_categories: Select the top protocol, service, and state categories.
        transform_categories: Transform the protocol, service, and state categories in the input dataframe.
        create_log1p_features: Create log1p features for the selected columns.
        preprocess: Preprocess the input dataframe.
        create_pool: Create a Pool object from the input dataframe.

    """

    def __init__(self, model_path : str):
        """
        Initialize the Preprocessor with the path to the trained model.

        Args:
            model_path (str): The path to the trained model.

        Returns:
            None

        """
        self.categorical_features = []
        self.numerical_features = []
        self.model = None
        self.top_prop_categories = None
        self.top_service_categories = None
        self.top_state_categories = None
        self.model_path = model_path

        if self.model_path:
            self.load_model_and_set_feature_names()
        self.selecting_categories()

    def load_model_and_set_feature_names(self):
        """
        Load the trained model and set the feature names.

        Args:
            None

        Returns:
            None

        """
        self.model = CatBoostClassifier()
        self.model.load_model(self.model_path)
        self.selected_features = self.model.feature_names_

    def feature_selection(self, df : pd.DataFrame):
        """
        Perform feature selection on the input dataframe.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:

            df (pd.DataFrame): The dataframe with selected features.

        """
        epsilon = 1e-10  # Small constant to avoid division by zero

        df["Speed of Operations to Speed of Data Bytes"] = np.log1p(df["sbytes"] / (df["dbytes"] + epsilon))
        df["Time for a Single Process"] = np.log1p(df["dur"] / (df["spkts"] + epsilon))
        df["Ratio of Data Flow"] = np.log1p(df["dbytes"] / (df["sbytes"] + epsilon))
        df["Ratio of Packet Flow"] = np.log1p(df["dpkts"] / (df["spkts"] + epsilon))
        df["Total Page Errors"] = np.log1p(df["dur"] * df["sloss"])
        df["Network Usage"] = np.log1p(df["sbytes"] + df["dbytes"])
        df["Network Activity Rate"] = np.log1p(df["spkts"] + df["dpkts"])

        df = df[self.selected_features] if hasattr(self, 'selected_features') else df
        return df

    def convert_data_types(self, df : pd.DataFrame):
        """
        Convert the data types of the columns in the input dataframe.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            df (pd.DataFrame): The dataframe with converted data types.

        """
        self.categorical_features = ['proto', 'service', 'state']
        self.numerical_features = [col for col in df.columns if col not in self.categorical_features]
        for column in self.categorical_features:
            df.loc[:, column] = df[column].astype('category')
        for column in self.numerical_features:
            df.loc[:, column] = df[column].astype(float)
        return df

    def selecting_categories(self):
        """
        Select the top protocol, service, and state categories.

        Args:
            None

        Returns:
            None

        """
        self.top_prop_categories = fe.top_prop_categories
        self.top_service_categories = fe.top_service_categories
        self.top_state_categories = fe.top_state_categories

    def transform_categories(self, df : pd.DataFrame):
        """
        Transform the protocol, service, and state categories in the input dataframe.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            df (pd.DataFrame): The dataframe with transformed categories.

        """
        df.loc[:, 'proto'] = np.where(df['proto'].isin(self.top_prop_categories), df['proto'], '-')
        df.loc[:, 'service'] = np.where(df['service'].isin(self.top_service_categories), df['service'], '-')
        df.loc[:, 'state'] = np.where(df['state'].isin(self.top_state_categories), df['state'], '-')
        return df

    def create_log1p_features(self, df : pd.DataFrame):
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
                # Convert to float before applying np.log1p to avoid dtype warnings
                df.loc[:, feature] = np.log1p(df[feature].astype(float)).astype('float32')
        return df

    def preprocess(self, df: pd.DataFrame):
        """
        Preprocess the input dataframe.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            df (pd.DataFrame): The preprocessed dataframe.

        """
        df = self.feature_selection(df)
        df = self.transform_categories(df)
        df = self.create_log1p_features(df)
        df = self.convert_data_types(df)
        return df

    def create_pool(self, X : pd.DataFrame):
        """
        Create a Pool object from the input dataframe.

        Args:
            X (pd.DataFrame): The input dataframe.

        Returns:
            Pool (Pool): The Pool object.

        """
        return Pool(
            data=X,
            cat_features=self.categorical_features
        )
