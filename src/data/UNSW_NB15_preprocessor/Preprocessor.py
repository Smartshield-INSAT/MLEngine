import pandas as pd
import numpy as np
import src.features.UNSW_NB15_features.feature_engineering as fe
from catboost import CatBoostClassifier, Pool
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class ModelPreprocessor:
    """
    Preprocessor class for the CatBoost model.
    """

    def __init__(self, model_path=None):
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
        self.model = CatBoostClassifier()
        self.model.load_model(self.model_path)
        self.selected_features = self.model.feature_names_

    def feature_selection(self, df):
        df = df[self.selected_features] if hasattr(self, 'selected_features') else df
        return df

    def convert_data_types(self, df):
        self.categorical_features = ['proto', 'service', 'state']
        self.numerical_features = [col for col in df.columns if col not in self.categorical_features]
        for column in self.categorical_features:
            df.loc[:, column] = df[column].astype('category')
        for column in self.numerical_features:
            df.loc[:, column] = df[column].astype(float)
        return df

    def selecting_categories(self):
        self.top_prop_categories = fe.top_prop_categories
        self.top_service_categories = fe.top_service_categories
        self.top_state_categories = fe.top_state_categories

    def transform_categories(self, df):
        df.loc[:, 'proto'] = np.where(df['proto'].isin(self.top_prop_categories), df['proto'], '-')
        df.loc[:, 'service'] = np.where(df['service'].isin(self.top_service_categories), df['service'], '-')
        df.loc[:, 'state'] = np.where(df['state'].isin(self.top_state_categories), df['state'], '-')
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
                # Convert to float before applying np.log1p to avoid dtype warnings
                df.loc[:, feature] = np.log1p(df[feature].astype(float)).astype('float32')
        return df
    def preprocess(self, df: pd.DataFrame):
        df = self.feature_selection(df)
        df = self.transform_categories(df)
        df = self.create_log1p_features(df)
        df = self.convert_data_types(df)
        return df

    def create_pool(self, X):
        return Pool(
            data=X,
            cat_features=self.categorical_features
        )
