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


class AdvancedFeatureEngineer:          # Advanced feature engineering pipeline
    
    def __init__(self, output_dir: str = "data/processed/", 
                 config_path: str = "data_config.yaml"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Always resolve config_path relative to the parent directory of this file unless absolute
        if not Path(config_path).is_absolute():
            # Get parent directory of this file (src/)
            parent_dir = Path(__file__).parent.parent
            self.config_path = str(parent_dir / "configs" / config_path)
        else:
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
                'methods': ['correlation', 'variance'],
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
        x = 0
        # 1. Remove highly correlated features
        if 'correlation' in self.config['feature_selection']['methods']:
            self.logger.info("Removing highly correlated features...")
            
            correlation_matrix = X_selected.corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            threshold = self.config['feature_selection']['correlation_threshold']
            high_corr_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > threshold)]
            print(f"Length of high correlated features:{len(high_corr_features)}")
            
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



    def create_feature_report(self, df_original: pd.DataFrame, df_processed: pd.DataFrame) -> Dict[str, Any]:       # Generate comprehensive feature engineering report
        
        self.logger.info("Generating feature engineering report...")
        
        report = {
            'summary': {
                'original_features': len(df_original.columns),
                'final_features': len(df_processed.columns),
                'features_added': len(df_processed.columns) - len(df_original.columns),
                'processing_timestamp': datetime.now().isoformat()
            },
            'data_quality': self.data_quality_report,
            'feature_metadata': self.feature_metadata,
            'feature_distribution': {},
            'preprocessing_artifacts': list(self.preprocessing_artifacts.keys())
        }
        
        numeric_features = df_processed.select_dtypes(include=[np.number]).columns
        categorical_features = df_processed.select_dtypes(include=['object', 'category']).columns
        
        
        report['feature_distribution'] = {          # Analyze feature distributions
            'numeric_features': len(numeric_features),
            'categorical_features': len(categorical_features),
            'binary_features': len([col for col in numeric_features 
                                  if df_processed[col].nunique() == 2])
        }
        
        if len(numeric_features) > 0:
            try:
                feature_importance = {}             # Quick feature importance using correlation with synthetic target
                for col in numeric_features[:20]:   # Limit to top 20 for performance
    
                    variance_score = df_processed[col].var()            # Create synthetic importance score based on variance and range
                    range_score = df_processed[col].max() - df_processed[col].min()
                    feature_importance[col] = variance_score * range_score
                
                sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)        # Sort by importance
                report['feature_importance_preview'] = dict(sorted_importance[:10])
                
            except Exception as e:
                self.logger.warning(f"Could not generate feature importance preview: {str(e)}")
        
        return report
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> str:          # Save processed data with metadata
        
        self.logger.info(f"Saving processed data to {filename}")
        
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        
        # Save metadata
        metadata = {
            'filename': filename,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'creation_timestamp': datetime.now().isoformat(),
            'preprocessing_artifacts': list(self.preprocessing_artifacts.keys())
        }
        
        metadata_path = self.output_dir / f"{filename.replace('.csv', '_metadata.json')}"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(filepath)
    
    def save_preprocessing_artifacts(self) -> str:                  # Save all preprocessing artifacts for later use
        artifacts_path = self.output_dir / "preprocessing_artifacts.pkl"
        joblib.dump(self.preprocessing_artifacts, artifacts_path)
        self.logger.info(f"Preprocessing artifacts saved to {artifacts_path}")
        return str(artifacts_path)
    
    def create_visualizations(self, df_original: pd.DataFrame, df_processed: pd.DataFrame,      # Create comprehensive visualizations for feature engineering analysis
                            target_col: str = None):
        
        self.logger.info("Creating feature engineering visualizations...")
        
        fig_dir = Path("reports/figures")           # Create figures directory
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Feature count comparison
        plt.figure(figsize=(10, 6))
        categories = ['Original Features', 'Final Features']
        counts = [len(df_original.columns), len(df_processed.columns)]
        colors = ['#3498db', '#2ecc71']
        
        bars = plt.bar(categories, counts, color=colors, alpha=0.8)
        plt.title('Feature Engineering: Before vs After', fontsize=16, fontweight='bold')
        plt.ylabel('Number of Features')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'feature_count_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Data types distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original data types
        original_types = df_original.dtypes.value_counts()
        ax1.pie(original_types.values, labels=original_types.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Original Data Types Distribution')
        
        # Processed data types
        processed_types = df_processed.dtypes.value_counts()
        ax2.pie(processed_types.values, labels=processed_types.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Processed Data Types Distribution')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'data_types_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Missing values before and after
        if df_original.isnull().sum().sum() > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Original missing values
            missing_original = df_original.isnull().sum()
            missing_original = missing_original[missing_original > 0]
            
            if len(missing_original) > 0:
                ax1.bar(range(len(missing_original)), missing_original.values)
                ax1.set_xticks(range(len(missing_original)))
                ax1.set_xticklabels(missing_original.index, rotation=45, ha='right')
                ax1.set_title('Missing Values - Original Data')
                ax1.set_ylabel('Missing Count')
            
            # Processed missing values
            missing_processed = df_processed.isnull().sum()
            missing_processed = missing_processed[missing_processed > 0]
            
            if len(missing_processed) > 0:
                ax2.bar(range(len(missing_processed)), missing_processed.values)
                ax2.set_xticks(range(len(missing_processed)))
                ax2.set_xticklabels(missing_processed.index, rotation=45, ha='right')
                ax2.set_title('Missing Values - Processed Data')
                ax2.set_ylabel('Missing Count')
            else:
                ax2.text(0.5, 0.5, 'No Missing Values', transform=ax2.transAxes,
                        ha='center', va='center', fontsize=16, fontweight='bold')
                ax2.set_title('Missing Values - Processed Data')
            
            plt.tight_layout()
            plt.savefig(fig_dir / 'missing_values_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Feature importance visualization (if business features were created)
        if 'business_features' in self.feature_metadata:
            business_features = self.feature_metadata['business_features']
            if business_features:
                plt.figure(figsize=(12, 8))
                
                # Calculate simple importance scores for business features
                importance_scores = {}
                for feature in business_features:
                    if feature in df_processed.columns:
                        # Use variance as importance proxy
                        importance_scores[feature] = df_processed[feature].var()
                
                if importance_scores:
                    features = list(importance_scores.keys())
                    scores = list(importance_scores.values())
                    
                    plt.barh(features, scores)
                    plt.title('Business Features - Variance-Based Importance', fontsize=14, fontweight='bold')
                    plt.xlabel('Variance Score')
                    plt.tight_layout()
                    plt.savefig(fig_dir / 'business_features_importance.png', dpi=300, bbox_inches='tight')
                    plt.close()
        
        # 5. Correlation heatmap for key features
        numeric_features = df_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 2:
            # Limit to most important features for visualization
            key_features = numeric_features[:15] if len(numeric_features) > 15 else numeric_features
            
            plt.figure(figsize=(12, 10))
            correlation_matrix = df_processed[key_features].corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
                       center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Feature Correlation Matrix (Top Features)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(fig_dir / 'feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info("Visualizations created successfully")


    def generate_summary_report(self, df_original: pd.DataFrame, df_processed: pd.DataFrame,        # Generate comprehensive HTML report
                               feature_report: Dict[str, Any], target_col: str = None) -> str:
        
        self.logger.info("Generating comprehensive feature engineering report...")
        
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TeleRetain: Phase 2 Feature Engineering Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
                h3 {{ color: #2c3e50; }}
                .metric {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeeba; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .info {{ background-color: #d1ecf1; border: 1px solid #bee5eb; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #fff3cd; padding: 20px; border-radius: 5px; margin: 15px 0; }}
                .feature-list {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                ul {{ margin: 10px 0; padding-left: 20px; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>TeleRetain: Phase 2 Feature Engineering Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>1. Executive Summary</h2>
                <div class="highlight">
                    <h3>Feature Engineering Results:</h3>
                    <ul>
                        <li><strong>Original Features:</strong> {feature_report['summary']['original_features']}</li>
                        <li><strong>Final Features:</strong> {feature_report['summary']['final_features']}</li>
                        <li><strong>Net Features Added:</strong> {feature_report['summary']['features_added']}</li>
                        <li><strong>Data Quality:</strong> {self._get_data_quality_summary()}</li>
                        <li><strong>Processing Status:</strong> ✅ Successfully Completed</li>
                    </ul>
                </div>
                
                <h2>2. Data Preprocessing Summary</h2>
                {self._generate_preprocessing_section()}
                
                <h2>3. Feature Engineering Details</h2>
                {self._generate_feature_engineering_section()}
                
                <h2>4. Feature Selection Results</h2>
                {self._generate_feature_selection_section()}
                
                <h2>5. Data Quality Assessment</h2>
                {self._generate_data_quality_section(df_original, df_processed)}
                
                <h2>6. Business Features Created</h2>
                {self._generate_business_features_section()}
                
                <h2>7. Next Steps & Recommendations</h2>
                {self._generate_recommendations_section()}
                
                <h2>8. Technical Artifacts</h2>
                <div class="info">
                    <h3>Saved Artifacts:</h3>
                    <ul>
                        <li>✅ Processed training data: <code>data/processed/train_processed.csv</code></li>
                        <li>✅ Preprocessing pipeline: <code>data/processed/preprocessing_artifacts.pkl</code></li>
                        <li>✅ Feature metadata: <code>data/processed/feature_metadata.json</code></li>
                        <li>✅ Visualizations: <code>reports/figures/</code></li>
                        <li>✅ Analysis logs: <code>logs/phase2_feature_engineering_*.log</code></li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save report
        report_path = Path("reports") / "feature_engineering_report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        self.logger.info(f"Comprehensive report saved to {report_path}")
        return str(report_path)
    
    def _get_data_quality_summary(self) -> str:             # Get data quality summary
        if 'missing_value_handling' in self.data_quality_report:
            return "Missing values handled ✅"
        return "No major issues detected ✅"
    
    def _generate_preprocessing_section(self) -> str:       # Generate preprocessing section of the report
        section = "<div class='metric'><h3>Preprocessing Steps Applied:</h3><ul>"
        
        if 'missing_value_handling' in self.data_quality_report:
            missing_info = self.data_quality_report['missing_value_handling']
            section += f"<li>✅ <strong>Missing Value Handling:</strong> {missing_info.get('numeric', {}).get('strategy', 'N/A')} for numeric, {missing_info.get('categorical', {}).get('strategy', 'N/A')} for categorical</li>"
        
        if 'outlier_handling' in self.data_quality_report:
            outlier_info = self.data_quality_report['outlier_handling']
            outlier_count = sum(info['outliers_detected'] for info in outlier_info.values())
            section += f"<li>✅ <strong>Outlier Handling:</strong> {outlier_count} outliers detected and handled</li>"
        
        if 'categorical_encoding' in self.data_quality_report:
            encoding_info = self.data_quality_report['categorical_encoding']
            section += f"<li>✅ <strong>Categorical Encoding:</strong> {encoding_info['method']} encoding applied</li>"
        
        if 'feature_scaling' in self.data_quality_report:
            scaling_info = self.data_quality_report['feature_scaling']
            section += f"<li>✅ <strong>Feature Scaling:</strong> {scaling_info['method']} scaling applied to {len(scaling_info['scaled_columns'])} features</li>"
        
        section += "</ul></div>"
        return section
    
    def _generate_feature_engineering_section(self) -> str:         # Generate feature engineering section
        
        section = ""
        
        if 'business_features' in self.feature_metadata:
            business_count = len(self.feature_metadata['business_features'])
            section += f"<div class='success'><h3>Business Features: {business_count} created</h3>"
            section += "<div class='feature-list'><strong>Features created:</strong><ul>"
            for feature in self.feature_metadata['business_features'][:10]:  # Show first 10
                section += f"<li>{feature}</li>"
            if business_count > 10:
                section += f"<li>... and {business_count - 10} more</li>"
            section += "</ul></div></div>"
        
        if 'interaction_features' in self.feature_metadata:
            interaction_count = len(self.feature_metadata['interaction_features'])
            section += f"<div class='info'><h3>Interaction Features: {interaction_count} created</h3></div>"
        
        return section
    
    def _generate_feature_selection_section(self) -> str:           # Generate feature selection section
        if 'feature_selection' not in self.feature_metadata:
            return "<div class='warning'>No feature selection applied</div>"
        
        selection_info = self.feature_metadata['feature_selection']
        section = "<div class='metric'><h3>Feature Selection Results:</h3><ul>"
        
        for method, info in selection_info.items():
            if 'removed_count' in info:
                section += f"<li><strong>{method.replace('_', ' ').title()}:</strong> {info['removed_count']} features removed</li>"
        
        section += "</ul></div>"
        return section
    
    def _generate_data_quality_section(self, df_original: pd.DataFrame, df_processed: pd.DataFrame) -> str:     # Generate data quality section
        section = "<div class='metric'><h3>Data Quality Metrics:</h3>"
        section += "<table><tr><th>Metric</th><th>Original</th><th>Processed</th><th>Status</th></tr>"
        
        # Missing values
        original_missing = df_original.isnull().sum().sum()
        processed_missing = df_processed.isnull().sum().sum()
        status = "✅ Resolved" if processed_missing == 0 else "⚠️ Some remaining"
        section += f"<tr><td>Missing Values</td><td>{original_missing:,}</td><td>{processed_missing:,}</td><td>{status}</td></tr>"
        
        # Data types
        original_numeric = len(df_original.select_dtypes(include=[np.number]).columns)
        processed_numeric = len(df_processed.select_dtypes(include=[np.number]).columns)
        section += f"<tr><td>Numeric Features</td><td>{original_numeric}</td><td>{processed_numeric}</td><td>✅</td></tr>"
        
        # Memory usage
        original_memory = df_original.memory_usage(deep=True).sum() / 1024**2
        processed_memory = df_processed.memory_usage(deep=True).sum() / 1024**2
        section += f"<tr><td>Memory Usage (MB)</td><td>{original_memory:.1f}</td><td>{processed_memory:.1f}</td><td>✅</td></tr>"
        
        section += "</table></div>"
        return section

    def _generate_business_features_section(self) -> str:           # Generate business features section
        if 'business_features' not in self.feature_metadata:
            return "<div class='warning'>No business features created</div>"
        
        business_features = self.feature_metadata['business_features']
        section = "<div class='success'><h3>Business-Specific Features Created:</h3>"
        section += "<div class='feature-list'><ul>"
        
        feature_descriptions = {
            'CLV_estimate': 'Customer Lifetime Value estimation (tenure × monthly charges)',
            'ARPU': 'Average Revenue Per User (total charges / tenure)',
            'total_services': 'Count of subscribed services',
            'premium_services': 'Count of premium services (streaming, tech support)',
            'contract_risk_score': 'Risk score based on contract type',
            'payment_risk_score': 'Risk score based on payment method',
            'customer_value_score': 'Combined customer value assessment'
        }
        
        for feature in business_features:
            description = feature_descriptions.get(feature, 'Domain-specific feature')
            section += f"<li><strong>{feature}:</strong> {description}</li>"
        
        section += "</ul></div></div>"
        return section
    

    def _generate_recommendations_section(self) -> str:         # Generate recommendations section
        
        recommendations = []
        
        # Data quality recommendations
        if 'missing_value_handling' in self.data_quality_report:
            recommendations.append("✅ Data preprocessing completed successfully")
        
        # Feature engineering recommendations  
        if 'business_features' in self.feature_metadata:
            recommendations.append("✅ Business features created - ready for model training")
        
        # Next steps
        recommendations.extend([
            "🎯 Proceed to Phase 3: Model Development and Training",
            "📊 Consider A/B testing different feature combinations",
            "🔄 Set up automated feature pipeline for production",
            "📈 Monitor feature performance and drift in production"
        ])
        
        section = "<div class='info'><h3>Recommendations & Next Steps:</h3><ul>"
        for rec in recommendations:
            section += f"<li>{rec}</li>"
        section += "</ul></div>"
        
        return section
    
    def run_complete_pipeline(self, data_path: str, target_col: str = 'Churn') -> str:      # Run the complete feature engineering pipeline
        
        self.logger.info("Starting complete feature engineering pipeline...")
        
        try:
            df_original = self.load_data(data_path)     # # Load data
            df = df_original.copy()
            
            df = self.handle_missing_values(df)         # Step 1: Handle missing values
            
            df = self.handle_outliers(df, target_col)   # Step 2: Handle outliers
            
            df = self.create_business_features(df)      # Step 3: Create business features
            
            df = self.create_interaction_features(df)   # Step 4: Create interaction features
            
            df = self.encode_categorical_variables(df, target_col)      # Step 5: Encode categorical variables
            
            df = self.scale_features(df, target_col)    # Step 6: Scale features
            
            if target_col in df.columns:                # Prepare for feature selection

                if df[target_col].dtype == 'object':    # Convert target to numeric if needed
                    y = (df[target_col] == self.config['positive_class']).astype(int)
                else:
                    y = df[target_col]
                
                X = df.drop(columns=[target_col])
                
                X_selected = self.feature_selection(X, y)   # Step 7: Feature selection
                
                df_processed = pd.concat([X_selected, y], axis=1)       # Combine with target
            else:
                df_processed = df
                
            feature_report = self.create_feature_report(df_original, df_processed)      # Generate comprehensive report
            
            self.create_visualizations(df_original, df_processed, target_col)           # Create visualizations
            
            self.save_processed_data(df_processed, 'train_processed.csv')               # Save processed data and artifacts
            self.save_preprocessing_artifacts()
            
            feature_report_path = self.output_dir / 'feature_engineering_report.json'   # Save feature report
            with open(feature_report_path, 'w') as f:
                json.dump(feature_report, f, indent=2, default=str)
            
            report_path = self.generate_summary_report(df_original, df_processed, feature_report, target_col)   # Generate HTML report
            
            self.logger.info("Feature engineering pipeline completed successfully!")
            self._print_pipeline_summary(df_original, df_processed)
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {str(e)}")
            raise

    def _print_pipeline_summary(self, df_original: pd.DataFrame, df_processed: pd.DataFrame):       # Print pipeline summary to console
   
        print("\n" + "-"*80)
        print("PHASE 2 FEATURE ENGINEERING SUMMARY")
        print("-"*80)
        
        print(f"Original Dataset: {df_original.shape[0]:,} rows × {df_original.shape[1]} columns")
        print(f"Processed Dataset: {df_processed.shape[0]:,} rows × {df_processed.shape[1]} columns")
        print(f"Features Added: {df_processed.shape[1] - df_original.shape[1]}")
        
        if 'business_features' in self.feature_metadata:
            business_count = len(self.feature_metadata['business_features'])
            print(f"Business Features Created: {business_count}")
        
        if 'interaction_features' in self.feature_metadata:
            interaction_count = len(self.feature_metadata['interaction_features'])
            print(f"Interaction Features Created: {interaction_count}")
        
        original_missing = df_original.isnull().sum().sum()         # # Data quality improvements
        processed_missing = df_processed.isnull().sum().sum()
        print(f"Missing Values: {original_missing:,} → {processed_missing:,}")
        
        print(f"\nArtifacts saved in: data/processed/")
        print(f"Report generated: reports/phase2_feature_engineering_report.html")
        print(f"Visualizations: reports/figures/")
        
        print("="*80)

def main():             # Main execution function
    
    # Configuration
    DATA_PATH = "data/raw/Telco-Customer-data.csv"
    TARGET_COLUMN = "Churn"
    OUTPUT_DIR = "data/processed/"
    
    feature_engineer = AdvancedFeatureEngineer(output_dir=OUTPUT_DIR)       # Create feature engineer instance
    
    report_path = feature_engineer.run_complete_pipeline(DATA_PATH, TARGET_COLUMN)  # Run complete pipeline
    
    print(f"\n Phase 2 completed successfully!")
    print(f" Report available at: {report_path}")

if __name__ == "__main__":
    main()
