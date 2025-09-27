
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, RFE, RFECV, 
    SelectFromModel, VarianceThreshold,
    chi2, f_classif, mutual_info_classif
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from typing import List, Dict, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

class AdvancedFeatureSelector:      # Advanced feature selection with multiple algorithms and validation
    
    def __init__(self, 
                 target_metric: str = 'roc_auc',
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
            target_metric: Metric to optimize for feature selection
            cv_folds: Number of cross-validation folds
        """
        self.target_metric = target_metric
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.feature_scores_ = {}
        self.selected_features_ = {}
        self.selection_history_ = {}
        
        self.logger = logging.getLogger(__name__)       # Setup logging
        
    def variance_threshold_selection(self, 
                                   X: pd.DataFrame, 
                                   threshold: float = 0.01) -> List[str]:       # Remove features with low variance
        """
        Args:
            X: Feature matrix
            threshold: Variance threshold
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Applying variance threshold selection (threshold={threshold})")
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        selected_features = X.columns[selector.get_support()].tolist()
        removed_features = X.columns[~selector.get_support()].tolist()
        
        self.feature_scores_['variance_threshold'] = dict(
            zip(X.columns, selector.variances_)
        )
        self.selected_features_['variance_threshold'] = selected_features
        
        self.logger.info(f"Variance threshold: kept {len(selected_features)}, removed {len(removed_features)}")
        return selected_features
    
    def correlation_selection(self, 
                            X: pd.DataFrame, 
                            threshold: float = 0.95) -> List[str]:  # Remove highly correlated features
        """
        Args:
            X: Feature matrix
            threshold: Correlation threshold
        """
        self.logger.info(f"Applying correlation-based selection (threshold={threshold})")
        
        corr_matrix = X.corr().abs()                # Calculate correlation matrix
        
        upper_triangle = corr_matrix.where(         # Find highly correlated pairs
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_remove = [col for col in upper_triangle.columns  # Identify features to remove
                    if any(upper_triangle[col] > threshold)]
        
        selected_features = [col for col in X.columns if col not in to_remove]
        
        self.selected_features_['correlation'] = selected_features
        self.selection_history_['correlation'] = {
            'removed_features': to_remove,
            'threshold': threshold,
            'high_correlation_pairs': []
        }
        
        high_corr_pairs = []            # Store high correlation pairs
        for col in to_remove:
            for row in corr_matrix.index:
                if corr_matrix.loc[row, col] > threshold and row != col:
                    high_corr_pairs.append((row, col, corr_matrix.loc[row, col]))
        
        self.selection_history_['correlation']['high_correlation_pairs'] = high_corr_pairs
        
        self.logger.info(f"Correlation selection: kept {len(selected_features)}, removed {len(to_remove)}")
        return selected_features
    
    def univariate_selection(self, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           method: str = 'f_classif',
                           k: int = 50) -> List[str]:       # Univariate feature selection
        """
        Args:
            X: Feature matrix
            y: Target vector
            method: Scoring method ('f_classif', 'chi2', 'mutual_info')
            k: Number of features to select

        """
        self.logger.info(f"Applying univariate selection ({method}, k={k})")
        
        if method == 'f_classif':       # Choose scoring function
            score_func = f_classif
        elif method == 'chi2':
            score_func = chi2
            X = X - X.min() + 1e-8      # Ensure all features are non-negative for chi2
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply selection
        k = min(k, X.shape[1])  # Ensure k doesn't exceed number of features
        selector = SelectKBest(score_func, k=k)
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store scores
        self.feature_scores_[f'univariate_{method}'] = dict(
            zip(X.columns, selector.scores_)
        )
        self.selected_features_[f'univariate_{method}'] = selected_features
        
        self.logger.info(f"Univariate selection ({method}): selected {len(selected_features)} features")
        return selected_features
    
    def recursive_feature_elimination(self, 
                                    X: pd.DataFrame, 
                                    y: pd.Series,
                                    estimator: Any = None,
                                    n_features: int = 30,
                                    cv: bool = True) -> List[str]:      # Recursive Feature Elimination (with optional CV)
        """
        Args:
            X: Feature matrix
            y: Target vector
            estimator: Base estimator for RFE
            n_features: Number of features to select
            cv: Whether to use cross-validation

        """
        method_name = 'rfe_cv' if cv else 'rfe'
        self.logger.info(f"Applying {method_name} (n_features={n_features})")
        
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
        
        n_features = min(n_features, X.shape[1])
        
        if cv:
            selector = RFECV(
                estimator=estimator,
                step=1,
                cv=self.cv_folds,
                scoring=self.target_metric,
                n_jobs=-1
            )
        else:
            selector = RFE(
                estimator=estimator,
                n_features_to_select=n_features,
                step=1
            )
        
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store rankings
        self.feature_scores_[method_name] = dict(
            zip(X.columns, selector.ranking_)
        )
        self.selected_features_[method_name] = selected_features
        
        if cv:
            self.selection_history_[method_name] = {
                'optimal_features': selector.n_features_,
                'cv_scores': selector.grid_scores_
            }
        
        self.logger.info(f"{method_name}: selected {len(selected_features)} features")
        return selected_features
    
    def model_based_selection(self, 
                            X: pd.DataFrame, 
                            y: pd.Series,
                            estimator_type: str = 'random_forest',
                            threshold: str = 'mean') -> List[str]:      # Model-based feature selection
        """
        Args:
            X: Feature matrix
            y: Target vector
            estimator_type: Type of estimator ('random_forest', 'extra_trees', 'lasso')
            threshold: Importance threshold ('mean', 'median', or float)
            
        """
        self.logger.info(f"Applying model-based selection ({estimator_type})")
        
        # Choose estimator
        if estimator_type == 'random_forest':
            estimator = RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
        elif estimator_type == 'extra_trees':
            estimator = ExtraTreesClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            )
        elif estimator_type == 'lasso':
            estimator = LassoCV(
                cv=self.cv_folds, 
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
        
        # Apply selection
        selector = SelectFromModel(estimator, threshold=threshold)
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Store feature importance/coefficients
        if hasattr(selector.estimator_, 'feature_importances_'):
            self.feature_scores_[f'model_based_{estimator_type}'] = dict(
                zip(X.columns, selector.estimator_.feature_importances_)
            )
        elif hasattr(selector.estimator_, 'coef_'):
            self.feature_scores_[f'model_based_{estimator_type}'] = dict(
                zip(X.columns, abs(selector.estimator_.coef_[0]))
            )
        
        self.selected_features_[f'model_based_{estimator_type}'] = selected_features
        
        self.logger.info(f"Model-based selection ({estimator_type}): selected {len(selected_features)} features")
        return selected_features
    
    def stability_selection(self, 
                           X: pd.DataFrame, 
                           y: pd.Series,
                           n_iterations: int = 100,
                           sample_fraction: float = 0.8,
                           threshold: float = 0.6) -> List[str]:        # Stability selection using bootstrap sampling
        """
        Args:
            X: Feature matrix
            y: Target vector
            n_iterations: Number of bootstrap iterations
            sample_fraction: Fraction of samples to use in each iteration
            threshold: Selection frequency threshold
            
        """
        self.logger.info(f"Applying stability selection ({n_iterations} iterations)")
        
        feature_selection_counts = np.zeros(X.shape[1])
        
        for i in range(n_iterations):
            # Bootstrap sampling
            sample_size = int(len(X) * sample_fraction)
            sample_indices = np.random.choice(len(X), sample_size, replace=True)
            
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
            
            # Feature selection using random forest
            rf = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state + i,
                n_jobs=-1
            )
            rf.fit(X_sample, y_sample)
            
            # Select top 50% features by importance
            importances = rf.feature_importances_
            threshold_value = np.percentile(importances, 50)
            selected_mask = importances >= threshold_value
            
            feature_selection_counts += selected_mask
        
        # Calculate selection frequencies
        selection_frequencies = feature_selection_counts / n_iterations
        stable_features_mask = selection_frequencies >= threshold
        
        selected_features = X.columns[stable_features_mask].tolist()
        
        self.feature_scores_['stability_selection'] = dict(
            zip(X.columns, selection_frequencies)
        )
        self.selected_features_['stability_selection'] = selected_features
        
        self.logger.info(f"Stability selection: selected {len(selected_features)} features")
        return selected_features
    
    def ensemble_selection(self, 
                          X: pd.DataFrame, 
                          y: pd.Series,
                          methods: List[str] = None,
                          voting_threshold: float = 0.5) -> List[str]:      # Ensemble feature selection combining multiple methods
        """

        Args:
            X: Feature matrix
            y: Target vector
            methods: List of methods to combine
            voting_threshold: Minimum fraction of methods that must select a feature
            
        """
        if methods is None:
            methods = ['univariate_f_classif', 'rfe', 'model_based_random_forest']
        
        self.logger.info(f"Applying ensemble selection with {len(methods)} methods")
        
        # Apply each method
        method_selections = {}
        
        for method in methods:
            if method == 'univariate_f_classif':
                features = self.univariate_selection(X, y, method='f_classif', k=50)
            elif method == 'rfe':
                features = self.recursive_feature_elimination(X, y, n_features=30, cv=False)
            elif method == 'model_based_random_forest':
                features = self.model_based_selection(X, y, estimator_type='random_forest')
            elif method == 'model_based_lasso':
                features = self.model_based_selection(X, y, estimator_type='lasso')
            else:
                self.logger.warning(f"Unknown method: {method}")
                continue
                
            method_selections[method] = set(features)
        
        # Count votes for each feature
        feature_votes = {}
        for feature in X.columns:
            votes = sum(1 for selection in method_selections.values() if feature in selection)
            feature