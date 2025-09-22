import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE, SelectKBest, chi2, mutual_info_classif
from typing import List, Dict, Any
from src.utils.logger import setup_logger
from src.utils.config import config

class FeatureEngineer:                      # Feature engineering utilities for the project

    def __init__(self):
        self.logger = setup_logger(__name__)
        self.data_config = config.data_config
        self.feature_names = []


    def create_business_features(self, df: pd.DataFrame) -> pd.DataFrame:       # create business specific features

        self.logger.info("Creating business features...")
        
        df_features = df.copy()

        if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
            df_features['CLV_estimate'] = df_features['tenure'] * df_features['MonthlyCharges']     # customer LTV approximation

        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df_features['ARPU'] = df_features['TotalCharges'] / (df_features['tenure'] + 1)         # ARPU approximation

        service_cols = [col for col in df.columns if 'Service' in col or 'Streaming' in col]
        if service_cols:
            df_features['total_services'] = df_features[service_cols].apply(                        # Service adoption count
                lambda x: (x == 'Yes').sum(), axis=1
            )

        if 'Contract' in df.columns:
            contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
            df_features['contract_risk_score'] = df_features['Contract'].map(contract_risk)          # Contract risk score

        if 'PaymentMethod' in df.columns:                               # payment method risk
            payment_risk = {
                'Electronic check': 3,
                'Mailed check': 2, 
                'Bank transfer (automatic)': 1,
                'Credit card (automatic)': 1
            }
            df_features['payment_risk_score'] = df_features['PaymentMethod'].map(payment_risk)
            
        self.logger.info("Business features created")
        return df_features    
    

    def create_interaction_features(self, df: pd.DataFrame, max_degree: int = 2) -> pd.DataFrame:       # create interaction features

        self.logger.info("Creating interaction features...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) > 10:                  # Limit to prevent feature explosion
            numeric_cols = numeric_cols[:10]

        poly = PolynomialFeatures(degree=max_degree, interaction_only=True, include_bias=False)     # only interaction terms and not bias term
        poly_features = poly.fit_transform(df[numeric_cols])

        feature_names = poly.get_feature_names_out(numeric_cols)
        df_poly = pd.DataFrame(poly_features, columns=feature_names, index=df.index)

        original_features = [col for col in feature_names if ' ' not in col]            # Remove original features to avoid duplication
        df_poly = df_poly.drop(columns=original_features)

        df_features = pd.concat([df, df_poly], axis=1)
        
        self.logger.info(f"Created {len(df_poly.columns)} interaction features")
        return df_features
    

    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:            # Create binned/discretized features

        self.logger.info("Creating binned features...")
        
        df_features = df.copy()

        if 'tenure' in df.columns:                  # Tenure bins
            df_features['tenure_group'] = pd.cut(
                df_features['tenure'],
                bins=[0, 12, 24, 36, 48, 100],
                labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4+yr']
            )

        if 'MonthlyCharges' in df.columns:          # monthly charge bins
            df_features['charges_group'] = pd.cut(
                df_features['MonthlyCharges'],
                bins=[0, 35, 65, 89, 200],
                labels=['Low', 'Medium', 'High', 'Premium']
            )

        self.logger.info("Binned features created")
        return df_features
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'rfe', n_features: int = 50) -> pd.DataFrame:        # feature selection

        self.logger.info(f"Selecting features using {method}...")
        
        if method == 'rfe':                         # Recursive feature elimination using Random forest model (while eliminatibg least important feature)
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
            
        elif method == 'chi2':                      # calculates a statistical relationshio between each feature and categorical 
            selector = SelectKBest(chi2, k=n_features)
            
        elif method == 'mutual_info':
            selector = SelectKBest(mutual_info_classif, k=n_features)
            
        else:
            self.logger.warning(f"Unknown selection method: {method}")
            return X
            
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        self.logger.info(f"Selected {len(selected_features)} features")
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)









