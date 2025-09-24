import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, chi2, mutual_info_classif, f_classif,
    RFE, RFECV, VarianceThreshold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
from typing import Dict, List, Tuple, Any, Optional
import os
import logging
from datetime import datetime
from pathlib import Path
import joblib
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class AdvancedfeatureEngineer:          # Advanced feature engineering pipeline
    
    def __init__(self, output_dir: str = "data/processed/", 
                 config_path: str = "configs/data_config.yaml"):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_path = config_path
        self.preprocessing_artifacts = {}
        self.feature_metadata = {}
        self.data_quality_report = {}
        
        self._setup_logging()           # Setup logging
        
        self._load_config()             # Load configuration if available


    def _setup_logging(self):           # Setup logging configuration
        
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/feature_engineering_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self):             # Load configuration from YAML file
        import yaml
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                self.logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:            # Get default configuration
        return {
            'target_column': 'Churn',
            'positive_class': 'Yes',
            'preprocessing': {
                'missing_value_strategy': {
                    'numeric': 'median',
                    'categorical': 'mode'
                },
                'outlier_detection': {
                    'method': 'iqr',
                    'iqr_multiplier': 1.5
                },
                'encoding': {
                    'categorical_method': 'onehot',
                    'handle_unknown': 'ignore',
                    'drop_first': True
                },
                'scaling': {
                    'method': 'standard'
                }
            },
            'feature_engineering': {
                'business_features': {
                    'create_clv': True,
                    'create_arpu': True,
                    'create_service_count': True,
                    'create_payment_risk': True
                },
                'interactions': {
                    'create': True,
                    'max_degree': 2
                },
                'binning': {
                    'tenure_bins': [0, 12, 24, 36, 48, 72],
                    'monthly_charges_bins': [0, 35, 65, 89, 120]
                }
            },
            'feature_selection': {
                'methods': ['correlation', 'variance', 'univariate'],
                'correlation_threshold': 0.95,
                'variance_threshold': 0.01,
                'univariate_k': 50
            }
        }
    
    def load_data(self, filepath: str) -> pd.DataFrame:     # Load data for feature engineering
        self.logger.info(f"Loading data from {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:      # Advanced missing value handling
        
        self.logger.info("Handling missing values with advanced techniques...")
        
        df_processed = df.copy()
        missing_info = {}
        
        missing_columns = df_processed.columns[df_processed.isnull().any()].tolist()        # Identify columns with missing values
        
        if not missing_columns:
            self.logger.info("No missing values found")
            return df_processed
            
        self.logger.info(f"Found missing values in columns: {missing_columns}")
        
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns          # Handle numeric missing values
        missing_numeric = [col for col in missing_columns if col in numeric_cols]
        
        if missing_numeric:
            strategy = self.config['preprocessing']['missing_value_strategy']['numeric']
            
            if strategy == 'knn':                       # Use KNN imputation for more sophisticated imputation
                imputer = KNNImputer(n_neighbors=5)
                df_processed[missing_numeric] = imputer.fit_transform(df_processed[missing_numeric])
                self.preprocessing_artifacts['knn_imputer'] = imputer
            
            else:                                                                       # Use simple imputation
                imputer = SimpleImputer(strategy=strategy)
                df_processed[missing_numeric] = imputer.fit_transform(df_processed[missing_numeric])
                self.preprocessing_artifacts['numeric_imputer'] = imputer
                
            missing_info['numeric'] = {
                'columns': missing_numeric,
                'strategy': strategy,
                'imputer_type': 'KNN' if strategy == 'knn' else 'Simple'
            }
        
        categorical_cols = df_processed.select_dtypes(include=['object']).columns       # Handle categorical missing values
        missing_categorical = [col for col in missing_columns if col in categorical_cols]
        
        if missing_categorical:
            strategy = self.config['preprocessing']['missing_value_strategy']['categorical']
            
            if strategy == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
                df_processed[missing_categorical] = imputer.fit_transform(df_processed[missing_categorical])
                self.preprocessing_artifacts['categorical_imputer'] = imputer
            elif strategy == 'unknown':
                df_processed[missing_categorical] = df_processed[missing_categorical].fillna('Unknown')
                
            missing_info['categorical'] = {
                'columns': missing_categorical,
                'strategy': strategy
            }
        
        self.data_quality_report['missing_value_handling'] = missing_info
        self.logger.info("Missing value handling completed")
        return df_processed
    

    def handle_outliers(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:        # Advanced outlier detection and handling
        
        self.logger.info("Detecting and handling outliers...")
        
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col and target_col in numeric_cols:       # Remove target column from outlier handling
            numeric_cols.remove(target_col)
            
        outlier_info = {}
        method = self.config['preprocessing']['outlier_detection']['method']
        
        for col in numeric_cols:
            original_count = len(df_processed)
            
            if method == 'iqr':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                multiplier = self.config['preprocessing']['outlier_detection']['iqr_multiplier']
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outliers_mask = (df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)  # Cap outliers instead of removing (better for ML)
                outlier_count = outliers_mask.sum()
                
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                
                outlier_info[col] = {
                    'method': 'IQR',
                    'outliers_detected': outlier_count,
                    'outlier_percentage': (outlier_count / original_count) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound},
                    'action': 'capped'
                }
                
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df_processed[col]))
                outliers_mask = z_scores > 3
                outlier_count = outliers_mask.sum()
                
                mean = df_processed[col].mean()     # Cap outliers at 3 standard deviations
                std = df_processed[col].std()
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                
                outlier_info[col] = {
                    'method': 'Z-Score',
                    'outliers_detected': outlier_count,
                    'outlier_percentage': (outlier_count / original_count) * 100,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound},
                    'action': 'capped'
                }
        
        self.data_quality_report['outlier_handling'] = outlier_info
        self.logger.info(f"Outlier handling completed using {method} method")
        return df_processed
    
    
    def create_business_features(self, df: pd.DataFrame) -> pd.DataFrame:           # Create domain-specific business features for telecom churn
        
        self.logger.info("Creating business-specific features...")
        
        df_features = df.copy()
        business_features = []
        
        # Customer Lifetime Value (CLV) estimation
        if (self.config['feature_engineering']['business_features']['create_clv'] and 
            'tenure' in df.columns and 'MonthlyCharges' in df.columns):
            
            df_features['CLV_estimate'] = df_features['tenure'] * df_features['MonthlyCharges']
            df_features['CLV_per_month'] = df_features['CLV_estimate'] / (df_features['tenure'] + 1)
            business_features.extend(['CLV_estimate', 'CLV_per_month'])
            
        # Average Revenue Per User (ARPU)
        if (self.config['feature_engineering']['business_features']['create_arpu'] and
            'TotalCharges' in df.columns and 'tenure' in df.columns):
            
            total_charges = pd.to_numeric(df_features['TotalCharges'], errors='coerce') # Handle string values in TotalCharges (common data quality issue)
            df_features['ARPU'] = total_charges / (df_features['tenure'] + 1)           # +1 to avoid division by zero
            df_features['ARPU'].fillna(df_features['MonthlyCharges'], inplace=True)     # Fallback to monthly charges
            business_features.append('ARPU')
            
        # Service adoption analysis
        if self.config['feature_engineering']['business_features']['create_service_count']:
            
            service_cols = [col for col in df.columns if any(service in col.lower() for service in      # Count of services subscribed
                           ['service', 'streaming', 'online', 'device', 'tech', 'multiple'])]
            
            if service_cols:
                df_features['total_services'] = 0
                df_features['premium_services'] = 0
                
                for col in service_cols:
                    df_features['total_services'] += (df_features[col] == 'Yes').astype(int)
                    
                    if any(premium in col.lower() for premium in ['streaming', 'tech', 'device', 'online']):        # Premium services (streaming, tech support, etc.)
                        df_features['premium_services'] += (df_features[col] == 'Yes').astype(int)
                        
                df_features['service_adoption_rate'] = df_features['total_services'] / len(service_cols)
                business_features.extend(['total_services', 'premium_services', 'service_adoption_rate'])
        
        # Contract and payment risk scoring
        if self.config['feature_engineering']['business_features']['create_payment_risk']:
            
            if 'Contract' in df.columns:            # Contract risk (month-to-month is highest risk)
                contract_risk_map = {
                    'Month-to-month': 3,
                    'One year': 2,
                    'Two year': 1
                }
                df_features['contract_risk_score'] = df_features['Contract'].map(contract_risk_map).fillna(2)
                business_features.append('contract_risk_score')
            
            if 'PaymentMethod' in df.columns:       # Payment method risk
                payment_risk_map = {
                    'Electronic check': 3,          # Highest risk
                    'Mailed check': 2,
                    'Bank transfer (automatic)': 1,     # Lowest risk
                    'Credit card (automatic)': 1
                }
                df_features['payment_risk_score'] = df_features['PaymentMethod'].map(payment_risk_map).fillna(2)
                business_features.append('payment_risk_score')
        
        # Customer tenure analysis
        if 'tenure' in df.columns:
           
            df_features['tenure_category'] = pd.cut(         # Tenure categories
                df_features['tenure'],
                bins=self.config['feature_engineering']['binning']['tenure_bins'],
                labels=['New', 'Short', 'Medium', 'Long', 'Loyal'],
                include_lowest=True
            )
            
            # Customer lifecycle stage
            df_features['is_new_customer'] = (df_features['tenure'] <= 6).astype(int)
            df_features['is_loyal_customer'] = (df_features['tenure'] >= 48).astype(int)
            business_features.extend(['is_new_customer', 'is_loyal_customer'])
        
        # Monthly charges analysis
        if 'MonthlyCharges' in df.columns:
            
            df_features['charges_category'] = pd.cut(               # Charges categories
                df_features['MonthlyCharges'],
                bins=self.config['feature_engineering']['binning']['monthly_charges_bins'],
                labels=['Low', 'Medium', 'High', 'Premium'],
                include_lowest=True
            )
            
            monthly_mean = df_features['MonthlyCharges'].mean()     # Price sensitivity indicators
            df_features['above_avg_charges'] = (df_features['MonthlyCharges'] > monthly_mean).astype(int)
            business_features.append('above_avg_charges')
        
        # Internet service analysis
        if 'InternetService' in df.columns:
            df_features['has_internet'] = (df_features['InternetService'] != 'No').astype(int)
            df_features['fiber_optic_user'] = (df_features['InternetService'] == 'Fiber optic').astype(int)
            business_features.extend(['has_internet', 'fiber_optic_user'])
        
        # Phone service analysis
        if 'PhoneService' in df.columns and 'MultipleLines' in df.columns:
            df_features['phone_service_complexity'] = 0
            df_features['phone_service_complexity'] += (df_features['PhoneService'] == 'Yes').astype(int)
            df_features['phone_service_complexity'] += (df_features['MultipleLines'] == 'Yes').astype(int)
            business_features.append('phone_service_complexity')
        
        # Demographics-based features
        if 'SeniorCitizen' in df.columns and 'Partner' in df.columns and 'Dependents' in df.columns:
            
            df_features['family_score'] = 0             # Family status score
            df_features['family_score'] += df_features['Partner'].eq('Yes').astype(int)
            df_features['family_score'] += df_features['Dependents'].eq('Yes').astype(int)
            
            df_features['life_stage'] = 'Single'        # Life stage categories
            df_features.loc[(df_features['Partner'] == 'Yes') & (df_features['Dependents'] == 'No'), 'life_stage'] = 'Couple'
            df_features.loc[df_features['Dependents'] == 'Yes', 'life_stage'] = 'Family'
            df_features.loc[df_features['SeniorCitizen'] == 1, 'life_stage'] = 'Senior'
            
            business_features.extend(['family_score'])
        
        # Customer value segmentation:  # Customer value score combining CLV and service adoption
        if 'CLV_estimate' in df_features.columns and 'total_services' in df_features.columns:
            
            df_features['customer_value_score'] = (
                df_features['CLV_estimate'] / df_features['CLV_estimate'].max() * 0.7 +
                df_features['total_services'] / df_features['total_services'].max() * 0.3
            )
            business_features.append('customer_value_score')
        
        self.feature_metadata['business_features'] = business_features
        self.logger.info(f"Created {len(business_features)} business features")
        return df_features
    
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:        # Create interaction features between important variables
        
        self.logger.info("Creating interaction features...")
        
        if not self.config['feature_engineering']['interactions']['create']:
            return df
            
        df_features = df.copy()
        interaction_features = []
        
        key_numeric = []                    # Key numeric features for interactions
        if 'tenure' in df.columns:
            key_numeric.append('tenure')
        if 'MonthlyCharges' in df.columns:
            key_numeric.append('MonthlyCharges')
        if 'CLV_estimate' in df_features.columns:
            key_numeric.append('CLV_estimate')
        if 'total_services' in df_features.columns:
            key_numeric.append('total_services')
            
        
        if len(key_numeric) >= 2:               # Create polynomial interactions (degree 2)
            from sklearn.preprocessing import PolynomialFeatures
            
            poly = PolynomialFeatures(
                degree=self.config['feature_engineering']['interactions']['max_degree'],
                interaction_only=True,
                include_bias=False
            )
            
            # Fit and transform
            poly_features = poly.fit_transform(df_features[key_numeric])
            feature_names = poly.get_feature_names_out(key_numeric)
            
            # Add new interaction features
            for i, feature_name in enumerate(feature_names):
                if ' ' in feature_name:                     # Only interaction terms, not original features
                    new_feature_name = feature_name.replace(' ', '_x_')
                    df_features[new_feature_name] = poly_features[:, i]
                    interaction_features.append(new_feature_name)
            
            self.preprocessing_artifacts['polynomial_features'] = poly
        
        # Manual important interactions
        interaction_pairs = [
            ('tenure', 'MonthlyCharges'),
            ('total_services', 'MonthlyCharges'),
            ('contract_risk_score', 'payment_risk_score'),
            ('family_score', 'total_services')
        ]
        
        for feat1, feat2 in interaction_pairs:
            if feat1 in df_features.columns and feat2 in df_features.columns:
                
                interaction_name = f"{feat1}_x_{feat2}"     # Multiplicative interaction
                df_features[interaction_name] = df_features[feat1] * df_features[feat2]
                interaction_features.append(interaction_name)
                
                if (df_features[feat1] > 0).all() and (df_features[feat2] > 0).all():       # Ratio interaction (if both positive)
                    ratio_name = f"{feat1}_div_{feat2}"
                    df_features[ratio_name] = df_features[feat1] / (df_features[feat2] + 1e-8)  # Small epsilon to avoid division by zero
                    interaction_features.append(ratio_name)
        
        self.feature_metadata['interaction_features'] = interaction_features
        self.logger.info(f"Created {len(interaction_features)} interaction features")
        return df_features
    
    
    def encode_categorical_variables(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:       # Advanced categorical encoding
        
        self.logger.info("Encoding categorical variables...")
        
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if target_col and target_col in categorical_cols:           # Remove target column from encoding
            categorical_cols.remove(target_col)
            
        encoding_info = {}
        encoding_method = self.config['preprocessing']['encoding']['categorical_method']
        
        if encoding_method == 'onehot':         # One-hot encoding
            encoded_df = pd.get_dummies(
                df_encoded,
                columns=categorical_cols,
                drop_first=self.config['preprocessing']['encoding']['drop_first'],
                prefix_sep='_'
            )
            
            new_columns = [col for col in encoded_df.columns if col not in df_encoded.columns]      # Track encoded columns
            encoding_info['method'] = 'onehot'
            encoding_info['original_columns'] = categorical_cols
            encoding_info['encoded_columns'] = new_columns
            
            df_encoded = encoded_df
            
        elif encoding_method == 'label':            # Label encoding
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                label_encoders[col] = le
                
            self.preprocessing_artifacts['label_encoders'] = label_encoders
            encoding_info['method'] = 'label'
            encoding_info['columns'] = categorical_cols
            
        elif encoding_method == 'target':
            # Target encoding (if target column is provided)
            if target_col and target_col in df_encoded.columns:
                target_encoders = {}
                
                # Convert target to numeric for encoding
                if df_encoded[target_col].dtype == 'object':
                    target_numeric = (df_encoded[target_col] == self.config['positive_class']).astype(int)
                else:
                    target_numeric = df_encoded[target_col]
                
                for col in categorical_cols:
                    
                    target_means = df_encoded.groupby(col)[target_col].apply(       # Calculate mean target value for each category
                        lambda x: (x == self.config['positive_class']).mean() if x.dtype == 'object' else x.mean()
                    )
                    
                    global_mean = target_numeric.mean()     # Apply smoothing to avoid overfitting
                    category_counts = df_encoded[col].value_counts()
                    
                    smoothing_factor = 10                   # Bayesian smoothing
                    smoothed_means = (target_means * category_counts + global_mean * smoothing_factor) / (category_counts + smoothing_factor)
                    
                    df_encoded[f'{col}_target_encoded'] = df_encoded[col].map(smoothed_means)
                    target_encoders[col] = smoothed_means.to_dict()
                    
                df_encoded.drop(columns=categorical_cols, inplace=True)     # Drop original categorical columns
                self.preprocessing_artifacts['target_encoders'] = target_encoders
                encoding_info['method'] = 'target'
                encoding_info['columns'] = categorical_cols
        
        self.data_quality_report['categorical_encoding'] = encoding_info
        self.logger.info(f"Categorical encoding completed using {encoding_method} method")
        return df_encoded
    

    def scale_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:         # Scale numerical features
        
        self.logger.info("Scaling numerical features...")
        
        df_scaled = df.copy()
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col and target_col in numeric_cols:           # Remove target column from scaling
            numeric_cols.remove(target_col)
        
        binary_cols = []                    # Remove binary encoded features from scaling (they're already 0/1)
        for col in numeric_cols:
            if df_scaled[col].nunique() == 2 and set(df_scaled[col].unique()).issubset({0, 1}):
                binary_cols.append(col)
        
        cols_to_scale = [col for col in numeric_cols if col not in binary_cols]
        
        scaling_method = self.config['preprocessing']['scaling']['method']
        
        if cols_to_scale:
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            else:
                self.logger.warning(f"Unknown scaling method: {scaling_method}. Using StandardScaler.")
                scaler = StandardScaler()
                
            df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
            self.preprocessing_artifacts['scaler'] = scaler
            
            scaling_info = {
                'method': scaling_method,
                'scaled_columns': cols_to_scale,
                'binary_columns_skipped': binary_cols
            }
            self.data_quality_report['feature_scaling'] = scaling_info
            
        self.logger.info(f"Feature scaling completed using {scaling_method}")
        return df_scaled
    

    def feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:         # Advanced feature selection using multiple techniques
        self.logger.info("Performing feature selection...")
        
        X_selected = X.copy()
        selection_info = {}
        selected_features = list(X.columns)
        
        # 1. Remove highly correlated features
        if 'correlation' in self.config['feature_selection']['methods']:
            self.logger.info("Removing highly correlated features...")
            
            correlation_matrix = X_selected.corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            threshold = self.config['feature_selection']['correlation_threshold']
            high_corr_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
            
            X_selected = X_selected.drop(columns=high_corr_features)
            selected_features = [f for f in selected_features if f not in high_corr_features]
            
            selection_info['correlation_removal'] = {
                'threshold': threshold,
                'removed_features': high_corr_features,
                'removed_count': len(high_corr_features)
            }
        
        # 2. Remove low variance features
        if 'variance' in self.config['feature_selection']['methods']:
            self.logger.info("Removing low variance features...")
            
            variance_threshold = self.config['feature_selection']['variance_threshold']
            selector = VarianceThreshold(threshold=variance_threshold)
            
            X_selected = pd.DataFrame(
                selector.fit_transform(X_selected),
                columns=X_selected.columns[selector.get_support()],
                index=X_selected.index
            )
            
            low_variance_features = X.columns[~selector.get_support()].tolist()
            selected_features = [f for f in selected_features if f not in low_variance_features]
            
            selection_info['variance_removal'] = {
                'threshold': variance_threshold,
                'removed_features': low_variance_features,
                'removed_count': len(low_variance_features)
            }
            
            self.preprocessing_artifacts['variance_selector'] = selector
        
        # 3. Univariate feature selection
        if 'univariate' in self.config['feature_selection']['methods']:
            self.logger.info("Performing univariate feature selection...")
            
            k = min(self.config['feature_selection']['univariate_k'], len(X_selected.columns))
            
            # Choose appropriate scoring function based on target type
            if y.dtype == 'object' or y.nunique() <= 10:
                # Classification
                if X_selected.select_dtypes(include=[np.number]).shape[1] == X_selected.shape[1]:       # All numeric features
                    selector = SelectKBest(f_classif, k=k)
                else:
                    selector = SelectKBest(mutual_info_classif, k=k)            # Mixed features, use mutual information
            else:
                selector = SelectKBest(f_classif, k=k)          # Regression (if target is continuous)
            
            X_selected = pd.DataFrame(
                selector.fit_transform(X_selected, y),
                columns=X_selected.columns[selector.get_support()],
                index=X_selected.index
            )
            
            removed_features = X.columns[~selector.get_support()].tolist()
            selected_features = [f for f in selected_features if f not in removed_features]
            
            # Get feature scores for ranking
            feature_scores = dict(zip(X.columns[selector.get_support()], selector.scores_[selector.get_support()]))
            
            selection_info['univariate_selection'] = {
                'k': k,
                'scoring_function': selector.score_func.__name__,
                'selected_features': list(X_selected.columns),
                'feature_scores': feature_scores,
                'removed_count': len(removed_features)
            }
            
            self.preprocessing_artifacts['univariate_selector'] = selector
        
        # 4. Recursive Feature Elimination (Optional)
        if 'rfe' in self.config['feature_selection']['methods']:
            self.logger.info("Performing Recursive Feature Elimination...")
            
            # Use RandomForest as estimator for feature importance
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            n_features = min(30, len(X_selected.columns))  # Limit to prevent overfitting
            
            selector = RFE(estimator, n_features_to_select=n_features, step=1)
            X_selected = pd.DataFrame(
                selector.fit_transform(X_selected, y),
                columns=X_selected.columns[selector.get_support()],
                index=X_selected.index
            )
            
            selection_info['rfe_selection'] = {
                'n_features_selected': n_features,
                'selected_features': list(X_selected.columns),
                'feature_ranking': dict(zip(X.columns, selector.ranking_))
            }
            
            self.preprocessing_artifacts['rfe_selector'] = selector
        
        self.feature_metadata['feature_selection'] = selection_info
        self.logger.info(f"Feature selection completed. Features: {len(X.columns)} → {len(X_selected.columns)}")
        return X_selected


