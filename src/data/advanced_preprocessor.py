import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from typing import List, Dict, Any, Optional, Union
import warnings

warnings.filterwarnings('ignore')

class AdvancedPreprocessor(BaseEstimator, TransformerMixin):        # advanced preprocessing pipeline with sophisticated techniques

    def __init__(self, 
                 imputation_strategy: str = 'knn',
                 scaling_method: str = 'robust',
                 handle_outliers: bool = True,
                 outlier_method: str = 'iqr',
                 create_polynomial_features: bool = False,
                 polynomial_degree: int = 2):

        self.imputation_strategy = imputation_strategy
        self.scaling_method = scaling_method
        self.handle_outliers = handle_outliers
        self.outlier_method = outlier_method
        self.create_polynomial_features = create_polynomial_features
        self.polynomial_degree = polynomial_degree
        
        # Initialize components
        self.imputer = None
        self.scaler = None
        self.outlier_detector = None
        self.polynomial_transformer = None
        self.feature_names_ = None
        self.numeric_features_ = None
        self.categorical_features_ = None

    def _identify_feature_types(self, X: pd.DataFrame) -> None:         # Identify numeric and categorical features
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def _setup_imputer(self, X: pd.DataFrame) -> None:              # Setup imputation strategy
        if self.imputation_strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        elif self.imputation_strategy == 'iterative':
            self.imputer = IterativeImputer(random_state=42, max_iter=10)
        else:
            from sklearn.impute import SimpleImputer
            self.imputer = SimpleImputer(strategy='median')
            
    def _setup_scaler(self) -> None:                        # Setup scaling method
        if self.scaling_method == 'robust':
            self.scaler = RobustScaler()
        elif self.scaling_method == 'quantile':
            self.scaler = QuantileTransformer(n_quantiles=1000, output_distribution='uniform')
        else:
            self.scaler = StandardScaler()
            
    def _setup_outlier_detector(self) -> None:          # "Setup outlier detection
        if self.handle_outliers and self.outlier_method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)

    def _handle_outliers_iqr(self, X: pd.DataFrame) -> pd.DataFrame:       # Handle outliers using IQR method
        X_processed = X.copy()
        
        for col in self.numeric_features_:
            if col in X_processed.columns:
                Q1 = X_processed[col].quantile(0.25)
                Q3 = X_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                X_processed[col] = X_processed[col].clip(lower=lower_bound, upper=upper_bound)  # Cap outliers instead of removing
                
        return X_processed
    
    def _create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:     # Create polynomial features

        if not self.create_polynomial_features:
            return X
            
        from sklearn.preprocessing import PolynomialFeatures
        
        if self.polynomial_transformer is None:
            self.polynomial_transformer = PolynomialFeatures(
                degree=self.polynomial_degree,
                interaction_only=False,
                include_bias=False
            )
            
        # Apply only to numeric features to prevent explosion
        numeric_data = X[self.numeric_features_[:5]]            # Limit to first 5 numeric features
        poly_features = self.polynomial_transformer.fit_transform(numeric_data)
        
        # Create DataFrame with polynomial features
        feature_names = self.polynomial_transformer.get_feature_names_out(numeric_data.columns)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=X.index)
        
        # Remove original features to avoid duplication
        original_features = [name for name in feature_names if ' ' not in name and '^' not in name]
        poly_df = poly_df.drop(columns=original_features, errors='ignore')
        
        # Combine with original data
        return pd.concat([X, poly_df], axis=1)
    
    def fit(self, X: pd.DataFrame, y=None):             # Fit the preprocessor
        
        self._identify_feature_types(X)                 # # Identify feature types
        
        self._setup_imputer(X)                          # Setup components
        self._setup_scaler()
        self._setup_outlier_detector()
        
        if self.numeric_features_:                      # Fit imputer on numeric features
            self.imputer.fit(X[self.numeric_features_])
            
        X_no_outliers = self._handle_outliers_iqr(X) if self.handle_outliers else X     # Handle outliers before fitting scaler
        
        if self.numeric_features_:                      # Fit scaler
            self.scaler.fit(X_no_outliers[self.numeric_features_])
            
        if self.outlier_detector is not None and self.numeric_features_:        # Fit outlier detector
            self.outlier_detector.fit(X_no_outliers[self.numeric_features_])
            
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:       # Transform the data
        X_processed = X.copy()
        
        if self.numeric_features_ and self.imputer is not None:     # Step 1: Impute missing values
            X_processed[self.numeric_features_] = self.imputer.transform(X_processed[self.numeric_features_])
            
        if self.handle_outliers:                            # Step 2: Handle outliers
            if self.outlier_method == 'iqr':
                X_processed = self._handle_outliers_iqr(X_processed)
            elif self.outlier_detector is not None:
                # Use isolation forest predictions to identify outliers
                outlier_predictions = self.outlier_detector.predict(X_processed[self.numeric_features_])
                X_processed['is_outlier'] = (outlier_predictions == -1).astype(int)     # For now, just mark outliers (in production, might want to handle differently)
                
        if self.numeric_features_ and self.scaler is not None:          # Step 3: Scale features
            X_processed[self.numeric_features_] = self.scaler.transform(X_processed[self.numeric_features_])
            
        X_processed = self._create_polynomial_features(X_processed)     # Step 4: Create polynomial features
        
        return X_processed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:       # Fit and transform the data
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):       # Get output feature names
        if input_features is None:
            return self.feature_names_
        return input_features


class BusinessFeatureCreator(BaseEstimator, TransformerMixin):          # Creates business-specific features for telecom churn prediction

    def __init__(self, 
                 create_clv: bool = True,
                 create_service_metrics: bool = True,
                 create_risk_scores: bool = True,
                 create_customer_segments: bool = True):
        
        self.create_clv = create_clv                            # create_clv: Create customer lifetime value features
        self.create_service_metrics = create_service_metrics    # create_service_metrics: Create service-related metrics
        self.create_risk_scores = create_risk_scores            # create_risk_scores: Create risk scoring features
        self.create_customer_segments = create_customer_segments    # create_customer_segments: Create customer segmentation features
        self.feature_names_ = []

    
    def _create_clv_features(self, X: pd.DataFrame) -> pd.DataFrame:        # Create Customer Lifetime Value features
        
        X_new = X.copy()
        
        if 'tenure' in X.columns and 'MonthlyCharges' in X.columns:
            X_new['CLV_estimate'] = X_new['tenure'] * X_new['MonthlyCharges']       # Basic CLV estimation
            
            X_new['CLV_per_month'] = X_new['CLV_estimate'] / (X_new['tenure'] + 1)  # CLV per month
            
            if 'TotalCharges' in X.columns:                                         # Revenue momentum
                total_charges = pd.to_numeric(X_new['TotalCharges'], errors='coerce')
                X_new['revenue_momentum'] = X_new['MonthlyCharges'] / (total_charges / (X_new['tenure'] + 1) + 1e-8)
                
            self.feature_names_.extend(['CLV_estimate', 'CLV_per_month', 'revenue_momentum'])
            
        return X_new
    

    def _create_service_metrics(self, X: pd.DataFrame) -> pd.DataFrame:     # Create service-related metrics
        
        X_new = X.copy()
        
        service_columns = [col for col in X.columns if                      # Identify service columns
                          any(service in col.lower() for service in 
                             ['service', 'streaming', 'online', 'device', 'tech'])]
        
        if service_columns:
            X_new['total_services'] = 0             # Total services count
            for col in service_columns:
                X_new['total_services'] += (X_new[col] == 'Yes').astype(int)
                
            X_new['service_adoption_rate'] = X_new['total_services'] / len(service_columns)     # Service adoption rate
            
            premium_services = [col for col in service_columns if           # Premium services (streaming, tech support)
                              any(premium in col.lower() for premium in ['streaming', 'tech', 'device'])]
            
            if premium_services:
                X_new['premium_services'] = 0
                for col in premium_services:
                    X_new['premium_services'] += (X_new[col] == 'Yes').astype(int)
                    
                X_new['premium_adoption_rate'] = X_new['premium_services'] / len(premium_services)
                self.feature_names_.extend(['premium_services', 'premium_adoption_rate'])
                
            self.feature_names_.extend(['total_services', 'service_adoption_rate'])
            
        return X_new
    
    def _create_risk_scores(self, X: pd.DataFrame) -> pd.DataFrame:     # Create risk scoring features
        
        X_new = X.copy()
        
        if 'Contract' in X.columns:                 # Contract risk score
            contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
            X_new['contract_risk_score'] = X_new['Contract'].map(contract_risk).fillna(2)
            self.feature_names_.append('contract_risk_score')
            
        if 'PaymentMethod' in X.columns:            # Payment method risk score
            payment_risk = {
                'Electronic check': 3,
                'Mailed check': 2,
                'Bank transfer (automatic)': 1,
                'Credit card (automatic)': 1
            }
            X_new['payment_risk_score'] = X_new['PaymentMethod'].map(payment_risk).fillna(2)
            self.feature_names_.append('payment_risk_score')
            
        if 'tenure' in X.columns:               # Tenure risk (new customers are higher risk)
            X_new['tenure_risk_score'] = np.where(X_new['tenure'] <= 6, 3,
                                        np.where(X_new['tenure'] <= 24, 2, 1))
            self.feature_names_.append('tenure_risk_score')
            
        return X_new
    
    def _create_customer_segments(self, X: pd.DataFrame) -> pd.DataFrame:       # Create customer segmentation features
        X_new = X.copy()
        
        if 'MonthlyCharges' in X.columns:               # Value-based segmentation
            
            monthly_charges = X_new['MonthlyCharges']   # Price tier
            X_new['price_tier'] = pd.cut(monthly_charges, 
                                       bins=[0, 35, 65, 89, float('inf')],
                                       labels=['Low', 'Medium', 'High', 'Premium'])
            
            avg_charges = monthly_charges.mean()        # Above average spender
            X_new['above_avg_spender'] = (monthly_charges > avg_charges).astype(int)
            self.feature_names_.append('above_avg_spender')
            
        if 'tenure' in X.columns:                       # Lifecycle stage
            tenure = X_new['tenure']
            X_new['lifecycle_stage'] = pd.cut(tenure,
                                            bins=[0, 6, 24, 48, float('inf')],
                                            labels=['New', 'Growing', 'Mature', 'Loyal'])
            
        if all(col in X.columns for col in ['Partner', 'Dependents']):      # Family status
            X_new['family_size_score'] = 0
            X_new['family_size_score'] += (X_new['Partner'] == 'Yes').astype(int)
            X_new['family_size_score'] += (X_new['Dependents'] == 'Yes').astype(int)
            self.feature_names_.append('family_size_score')
            
        return X_new
    
    def fit(self, X: pd.DataFrame, y=None):                 # Fit the feature creator (no-op for this transformer)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:   # Create business features
        X_transformed = X.copy()
        self.feature_names_ = []
        
        if self.create_clv:
            X_transformed = self._create_clv_features(X_transformed)
            
        if self.create_service_metrics:
            X_transformed = self._create_service_metrics(X_transformed)
            
        if self.create_risk_scores:
            X_transformed = self._create_risk_scores(X_transformed)
            
        if self.create_customer_segments:
            X_transformed = self._create_customer_segments(X_transformed)
            
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:       # Fit and transform the data
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):                   # Get output feature names
        return self.feature_names_
    

class FeatureInteractionCreator(BaseEstimator, TransformerMixin):           # Creates interaction features between important variables
    
    def __init__(self, 
                 interaction_degree: int = 2,
                 max_features: int = 10,
                 include_polynomial: bool = True):          # Initialize Feature Interaction Creator
        
        self.interaction_degree = interaction_degree
        self.max_features = max_features
        self.include_polynomial = include_polynomial
        self.poly_transformer = None
        self.feature_names_ = []
        self.important_features_ = []

    def _identify_important_features(self, X: pd.DataFrame) -> List[str]:       # Identify most important features for interaction
        
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        feature_importance = {}             # Use variance as a simple importance measure
        for col in numeric_features:
            if X[col].var() > 0:            # Only features with variance
                feature_importance[col] = X[col].var()
                
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)      # Sort by importance and take top features
        return [feature[0] for feature in sorted_features[:self.max_features]]
    
    def fit(self, X: pd.DataFrame, y=None):     # Fit the interaction creator
        
        self.important_features_ = self._identify_important_features(X)
        
        if self.include_polynomial and len(self.important_features_) > 1:
            from sklearn.preprocessing import PolynomialFeatures
            self.poly_transformer = PolynomialFeatures(
                degree=self.interaction_degree,
                interaction_only=True,
                include_bias=False
            )
            
            if self.important_features_:            # Fit on important features only
                self.poly_transformer.fit(X[self.important_features_])
                
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:       # Create interaction features
        X_transformed = X.copy()
        self.feature_names_ = []
        
        if self.poly_transformer is not None and self.important_features_:      # Create polynomial interactions
            poly_features = self.poly_transformer.transform(X[self.important_features_])
            feature_names = self.poly_transformer.get_feature_names_out(self.important_features_)
            
            for i, name in enumerate(feature_names):        # Add only interaction terms (not original features)
                if ' ' in name:                             # Interaction term
                    new_name = name.replace(' ', '_x_')
                    X_transformed[new_name] = poly_features[:, i]
                    self.feature_names_.append(new_name)
                    
        important_pairs = [                     # Create manual important interactions
            ('tenure', 'MonthlyCharges'),
            ('tenure', 'total_services'),
            ('MonthlyCharges', 'total_services'),
            ('contract_risk_score', 'payment_risk_score')
        ]
        
        for feat1, feat2 in important_pairs:
            if feat1 in X_transformed.columns and feat2 in X_transformed.columns:
                # Multiplicative interaction
                interaction_name = f"{feat1}_mult_{feat2}"
                X_transformed[interaction_name] = X_transformed[feat1] * X_transformed[feat2]
                self.feature_names_.append(interaction_name)
                
                # Ratio interaction
                if (X_transformed[feat2] != 0).all():
                    ratio_name = f"{feat1}_div_{feat2}"
                    X_transformed[ratio_name] = X_transformed[feat1] / (X_transformed[feat2] + 1e-8)
                    self.feature_names_.append(ratio_name)
                    
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:       # Fit and transform the data
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):                   # Get output feature names
        return self.feature_names_


