"""
Advanced Model Evaluation Metrics
=================================

This module provides comprehensive model evaluation metrics specifically
designed for customer churn prediction with business impact analysis.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, log_loss, brier_score_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings

warnings.filterwarnings('ignore')

class ChurnMetricsCalculator:
    """
    Specialized metrics calculator for customer churn prediction
    """
    
    def __init__(self, cost_matrix: Optional[Dict[str, float]] = None):
        """
        Initialize metrics calculator
        
        Args:
            cost_matrix: Dictionary with cost parameters for business metrics
                        {'retention_cost', 'churn_cost', 'avg_clv'}
        """
        self.cost_matrix = cost_matrix or {
            'retention_cost': 50,    # Cost to retain a customer
            'churn_cost': 500,       # Cost of losing a customer
            'avg_clv': 1200          # Average Customer Lifetime Value
        }

    def calculate_comprehensive_metrics(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray,
                                      y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities for positive class
            
        Returns:
            Dictionary with all calculated metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._basic_classification_metrics(y_true, y_pred, y_pred_proba))
        
        # Advanced classification metrics
        metrics.update(self._advanced_classification_metrics(y_true, y_pred, y_pred_proba))
        
        # Business-specific metrics
        metrics.update(self._business_metrics(y_true, y_pred, y_pred_proba))
        
        # Ranking and probability calibration metrics
        metrics.update(self._ranking_metrics(y_true, y_pred_proba))
        
        return metrics
    
    def _basic_classification_metrics(self, y_true: np.ndarray, 
                                    y_pred: np.ndarray,
                                    y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba)
        }
    
    def _advanced_classification_metrics(self, y_true: np.ndarray,
                                       y_pred: np.ndarray, 
                                       y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate advanced classification metrics"""
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Advanced metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Precision-Recall AUC
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        # Balanced accuracy
        balanced_acc = (sensitivity + specificity) / 2
        
        # Matthews Correlation Coefficient
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den != 0 else 0
        
        # Brier Score (probability calibration)
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        return {
            'specificity': specificity,
            'sensitivity': sensitivity,
            'balanced_accuracy': balanced_acc,
            'precision_recall_auc': pr_auc,
            'matthews_corr_coef': mcc,
            'brier_score': brier_score,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
    

    def _business_metrics(self, y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate business-specific metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Cost calculations
        retention_cost = self.cost_matrix['retention_cost']
        churn_cost = self.cost_matrix['churn_cost']
        avg_clv = self.cost_matrix['avg_clv']
        
        # Business metrics
        total_cost = (fp * retention_cost) + (fn * churn_cost)
        total_customers = len(y_true)
        cost_per_customer = total_cost / total_customers
        
        # Revenue calculations
        revenue_saved = tp * avg_clv  # Successfully retained customers
        revenue_lost = fn * avg_clv   # Missed churning customers
        
        # Efficiency metrics
        churn_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # ROI calculation
        retention_investment = (tp + fp) * retention_cost
        roi = (revenue_saved - retention_investment) / retention_investment if retention_investment > 0 else 0
        
        return {
            'customers_at_risk_identified': int(tp),
            'customers_missed': int(fn),
            'false_alarms': int(fp),
            'churn_detection_rate': churn_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'total_cost': total_cost,
            'cost_per_customer': cost_per_customer,
            'revenue_saved': revenue_saved,
            'revenue_lost': revenue_lost,
            'roi_percentage': roi * 100,
            'net_benefit': revenue_saved - retention_investment
        }