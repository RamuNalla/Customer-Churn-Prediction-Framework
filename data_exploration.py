import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime
import logging
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataExplorer:                     # comprehensive data exploration class

    def __init__(self, data_path: str, output_dir: str = "reports/"):
        self.data_path = data_path
        self.output_dir = output_dir    
        self.df = None
        self.data_qualitty_report = {}
        self.business_insights = {}

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/figures", exist_ok=True)

        self._setup_logging()               # run this function immediately after an object is created


    def _setup_logging(self):   
        logging.basicConfig(                # main function that configures the root logger
            level=logging.INFO,             # set the logging level to INFO (captures INFO, WARNING, ERROR, CRITICAL)
            format = '%(asctime)s - %(levelname)s - %(message)s',       # log message format (timestamp, level, message)
            handlers=[
                logging.FileHandler(f"{self.output_dir}/data_exploration.log"),     # log messages to a file
                logging.StreamHandler()                                             # also print log messages to the console
            ]
        )
        self.logger = logging.getLogger(__name__)       # it helps include the name of the module where the logger is used during logging


    def load_data(self) -> pd.DataFrame:                # load and perform initial data validation
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            
            self.logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            
            self.data_quality_report['shape'] = self.df.shape
            self.data_quality_report['columns'] = list(self.df.columns)
            self.data_quality_report['dtypes'] = self.df.dtypes.to_dict()

            return self.df
        
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    
    def data_quality_assessment(self) -> Dict[str, Any]:          # assess data quality 
        
        self.logger.info("Starting data quality assessment...  ")

        self.data_quality_report['basic_stats'] = {                 # basic statistics about the dataset
            'total_records': len(self.df),
            'total_features': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2      # Total memory usage by DF in MB
        } 

        missing_data = self.df.isnull().sum()                           # count of missing values per column
        missing_percentage = (missing_data / len(self.df)) * 100        # percentage of missing values per column

        self.data_quality_report['missing_values'] = {
            'columns_with_missing': missing_data[missing_data > 0].to_dict(),
            'missing_percentage': missing_percentage[missing_percentage > 0].to_dict(),
            'total_missing_cells': int(missing_data.sum()),
            'percentage_missing_cells': (missing_data.sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        }

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        self.data_quality_report['data_types'] = {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'numeric_count': len(numeric_cols),
            'categorical_count': len(categorical_cols)
        }

        duplicates = self.df.duplicated().sum()               # count of duplicate rows
        self.data_quality_report['duplicates'] = {
            'duplicate_count': duplicates,
            'duplicate_percentage': (duplicates / len(self.df)) * 100
        }

        unique_values = {}                              # count and percentage of unique values per column  
        for col in self.df.columns:
            unique_values[col] = {
                'unique_count': self.df[col].nunique(),
                'unique_percentage': (self.df[col].nunique() / len(self.df)) * 100
            }
        
        self.data_quality_report['unique_values'] = unique_values

        issues = []

        all_missing = [col for col in self.df.columns if self.df[col].isnull().all()]
        if all_missing:
            issues.append(f"Columns with all missing values: {all_missing}")

        high_cardinality = [col for col in categorical_cols if self.df[col].nunique() > 50]
        if high_cardinality:
            issues.append(f"High cardinality categorical columns: {high_cardinality}")
        
        self.data_quality_report['potential_issues'] = issues
        
        self.logger.info("Data quality assessment completed")
        return self.data_quality_report

    
    def outlier_detection(self) -> Dict[str, Any]:              # detect outliers in numberic columns using multiple methods
        
        self.logger.info("Starting outlier detection...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if 'customerID' in numeric_cols:                        # Remove ID column if numeric
            numeric_cols.remove('customerID')

        outlier_report = {}

        for col in numeric_cols:
            col_data = self.df[col].dropna()
            
            Q1 = col_data.quantile(0.25)             # IQR Method
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

            z_scores = np.abs(stats.zscore(col_data))       # Z-Score Method
            zscore_outliers = (z_scores > 3).sum()

            median = np.median(col_data)                    # Modified Z-Score Method
            mad = np.median(np.abs(col_data - median))
            modified_z_scores = 0.6745 * (col_data - median) / mad
            modified_zscore_outliers = (np.abs(modified_z_scores) > 3.5).sum()

            outlier_report[col] = {
                'iqr_outliers': iqr_outliers,
                'iqr_percentage': (iqr_outliers / len(col_data)) * 100,
                'zscore_outliers': zscore_outliers,
                'zscore_percentage': (zscore_outliers / len(col_data)) * 100,
                'modified_zscore_outliers': modified_zscore_outliers,
                'modified_zscore_percentage': (modified_zscore_outliers / len(col_data)) * 100,
                'bounds': {
                    'iqr_lower': lower_bound,
                    'iqr_upper': upper_bound
                }
            }

        self.data_quality_report['outliers'] = outlier_report
        self.logger.info("Outlier detection completed")
        return outlier_report
    

    def target_variable_analysis(self, target_col: str = 'Churn') -> Dict[str, Any]:        # analyze the target variable distribution and its characteristics

        self.logger.info(f"Analyzing target variable: {target_col}")

        if target_col not in self.df.columns:
            self.logger.error(f"Target column '{target_col}' not found in dataset")
            return {}
        
        target_analysis = {}

        value_counts = self.df[target_col].value_counts()                       # basic distribution
        percentage = self.df[target_col].value_counts(normalize=True) * 100 

        target_analysis['distribution'] = {
            'value_counts': value_counts.to_dict(),
            'percentages': percentage.to_dict(),
            'total_samples': len(self.df[target_col])
        }

        if len(value_counts) == 2:              # class imbalance analysis for Binary classification
            minority_class = value_counts.min()
            majority_class = value_counts.max()
            imbalance_ratio = majority_class / minority_class
            
            target_analysis['class_balance'] = {
                'minority_class_count': minority_class,
                'majority_class_count': majority_class,
                'imbalance_ratio': imbalance_ratio,
                'is_imbalanced': imbalance_ratio > 1.5
            }
        
        missing_target = self.df[target_col].isnull().sum()         # missing values in target column
        target_analysis['missing_values'] = {
            'count': missing_target,
            'percentage': (missing_target / len(self.df)) * 100
        }

        self.data_quality_report['target_analysis'] = target_analysis
        self.logger.info("Target variable analysis completed")
        return target_analysis

    
    def correlation_analysis(self) -> Dict[str, Any]:   # perform correlation analysis for numeric features

        self.logger.info("Starting correlation analysis...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            self.logger.warning("Insufficient numeric columns for correlation analysis")
            return {}

        corr_matrix = self.df[numeric_cols].corr()          # calculate correlation matrix  
        
        high_corr_pairs = []                                # identify highly correlated pairs
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # Threshold for high correlation
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        correlation_analysis = {
            'correlation_matrix': corr_matrix.to_dict(),
            'high_correlations': high_corr_pairs,
            'max_correlation': corr_matrix.abs().max().max(),
            'mean_correlation': corr_matrix.abs().mean().mean()
        }

        self.data_quality_report['correlation_analysis'] = correlation_analysis
        self.logger.info("Correlation analysis completed")
        return correlation_analysis
    
    def categorical_feature_analysis(self) -> Dict[str, Any]:     # analyze categorical analysis results

        self.logger.info("Starting categorical analysis...") 

        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        categorical_analysis = {}

        for col in categorical_cols:
            value_counts = self.df[col].value_counts()

            categorical_analysis[col] = {
                'unique_values': self.df[col].nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'most_frequent_percentage': (value_counts.iloc[0] / len(self.df)) * 100 if len(value_counts) > 0 else 0,
                'value_distribution': value_counts.head(10).to_dict(),
                'is_high_cardinality': self.df[col].nunique() > 20
            }

        self.data_quality_report['categorical_analysis'] = categorical_analysis
        self.logger.info("Categorical analysis completed")
        return categorical_analysis

    def business_insights_generation(self, target_col: str = 'Churn') -> Dict[str, Any]:        # generate business insights from the data
        
        self.logger.info("Generating business insights ...")

        insights = {}

        if target_col in self.df.columns:
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            
            if target_col in categorical_cols:
                categorical_cols.remove(target_col)

            churn_by_segment = {}
            
            for col in categorical_cols[:5]:        # Limit to top 5 categorical columns
                if self.df[col].nunique() <= 10:    # Only for low cardinality columns
                    churn_rate = self.df.groupby(col)[target_col].apply(
                        lambda x: (x == 'Yes').mean() * 100 if 'Yes' in x.values else 0
                    )
                    churn_by_segment[col] = churn_rate.to_dict()
            
            insights['churn_by_segment'] = churn_by_segment


            if 'MonthlyCharges' in self.df.columns:         # Revenue impact analysis (if applicable)
                avg_monthly_charges = self.df.groupby(target_col)['MonthlyCharges'].mean()
                insights['revenue_impact'] = {
                    'avg_monthly_charges_by_churn': avg_monthly_charges.to_dict(),
                    'total_monthly_revenue_at_risk': (
                        self.df[self.df[target_col] == 'Yes']['MonthlyCharges'].sum()
                        if 'Yes' in self.df[target_col].values else 0
                    )
                }

            if 'tenure' in self.df.columns:             # Tenure analysis
                tenure_churn = self.df.groupby(pd.cut(self.df['tenure'], bins=5))[target_col].apply(
                    lambda x: (x == 'Yes').mean() * 100 if 'Yes' in x.values else 0
                )
                insights['tenure_analysis'] = {
                    'churn_by_tenure_bins': tenure_churn.to_dict()
                }

        insights['data_completeness'] = {
            'overall_completeness': ((self.df.notna().sum().sum()) / (len(self.df) * len(self.df.columns))) * 100,
            'columns_with_missing_data': len([col for col in self.df.columns if self.df[col].isnull().sum() > 0]),
            'records_with_missing_data': len(self.df[self.df.isnull().any(axis=1)])
        }

        self.business_insights = insights
        self.logger.info("Business insights generation completed")
        return insights
    
    def create_visualizations(self):            # create comprehensive visualizations for the dataset

        self.logger.info("Creating visualizations...")
        
        plt.rcParams['figure.figsize'] = (12, 8)        # plotting style (runtime configuration)

        if 'Churn' in self.df.columns:                  # plot-1: Target variable distribution
            plt.figure(figsize=(10, 6))
            churn_counts = self.df['Churn'].value_counts()
            colors = ['#2E86AB', '#A23B72']
            
            plt.subplot(1, 2, 1)
            churn_counts.plot(kind='bar', color=colors)
            plt.title('Churn Distribution (Count)', fontsize=14, fontweight='bold')
            plt.xlabel('Churn Status')
            plt.ylabel('Count')
            plt.xticks(rotation=0)
            
            plt.subplot(1, 2, 2)
            churn_pct = self.df['Churn'].value_counts(normalize=True) * 100
            plt.pie(churn_pct.values, labels=churn_pct.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
            plt.title('Churn Distribution (Percentage)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/figures/churn_distribution.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()


        if self.df.isnull().sum().sum() > 0:            # plot-2: Missing values heatmap
            plt.figure(figsize=(12, 8))
            missing_data = self.df.isnull()
            
            if missing_data.sum().sum() > 0:
                sns.heatmap(missing_data, cbar=True, yticklabels=False, 
                           cmap='viridis', xticklabels=True)
                plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/figures/missing_values_heatmap.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()

        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()   # plot-3: Correlation matrix for numeric features  
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.df[numeric_cols].corr()
            
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r',
                       center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Correlation Matrix (Numeric Variables)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/figures/correlation_matrix.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

        
        numeric_cols = [col for col in numeric_cols if col != 'customerID']     # plot-4: Distribution of numeric features
        if numeric_cols:
            fig, axes = plt.subplots(nrows=(len(numeric_cols)+2)//3, ncols=3, 
                                   figsize=(15, 5*((len(numeric_cols)+2)//3)))
            axes = axes.ravel() if len(numeric_cols) > 1 else [axes]
            
            for idx, col in enumerate(numeric_cols):
                if idx < len(axes):
                    self.df[col].hist(bins=30, ax=axes[idx], alpha=0.7, color='skyblue')
                    axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Frequency')
            
            for idx in range(len(numeric_cols), len(axes)):
                axes[idx].set_visible(False)
                
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/figures/numeric_distributions.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()       # plot-5: Distribution of categorical features
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        cat_cols_viz = categorical_cols[:6]         # Limit to first 6 categorical columns for visualization
        
        if cat_cols_viz:
            fig, axes = plt.subplots(nrows=(len(cat_cols_viz)+2)//3, ncols=3, 
                                   figsize=(15, 5*((len(cat_cols_viz)+2)//3)))
            axes = axes.ravel() if len(cat_cols_viz) > 1 else [axes]
            
            for idx, col in enumerate(cat_cols_viz):
                if idx < len(axes) and self.df[col].nunique() <= 10:
                    value_counts = self.df[col].value_counts()
                    axes[idx].bar(range(len(value_counts)), value_counts.values, 
                                color='lightcoral', alpha=0.7)
                    axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                    axes[idx].set_xlabel(col)
                    axes[idx].set_ylabel('Count')
                    axes[idx].set_xticks(range(len(value_counts)))
                    axes[idx].set_xticklabels(value_counts.index, rotation=45)
            
            for idx in range(len(cat_cols_viz), len(axes)):         # Hide empty subplots
                axes[idx].set_visible(False)
                
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/figures/categorical_distributions.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()


        if 'Churn' in self.df.columns:
            categorical_analysis_cols = [col for col in categorical_cols if self.df[col].nunique() <= 8]            # plot-6: churn by categories
            
            if categorical_analysis_cols:
                fig, axes = plt.subplots(nrows=(len(categorical_analysis_cols)+2)//3, ncols=3,
                                       figsize=(15, 5*((len(categorical_analysis_cols)+2)//3)))
                axes = axes.ravel() if len(categorical_analysis_cols) > 1 else [axes]
                
                for idx, col in enumerate(categorical_analysis_cols):
                    if idx < len(axes):
                        churn_rate = self.df.groupby(col)['Churn'].apply(
                            lambda x: (x == 'Yes').mean() * 100
                        )
                        
                        axes[idx].bar(range(len(churn_rate)), churn_rate.values, 
                                    color='orange', alpha=0.7)
                        axes[idx].set_title(f'Churn Rate by {col}', fontweight='bold')
                        axes[idx].set_xlabel(col)
                        axes[idx].set_ylabel('Churn Rate (%)')
                        axes[idx].set_xticks(range(len(churn_rate)))
                        axes[idx].set_xticklabels(churn_rate.index, rotation=45)
                
                for idx in range(len(categorical_analysis_cols), len(axes)):
                    axes[idx].set_visible(False)
                    
                plt.tight_layout()
                plt.savefig(f'{self.output_dir}/figures/churn_rate_by_categories.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()


        if 'Churn' in self.df.columns and numeric_cols:         # plot-7: Numeric features by churn status
            fig, axes = plt.subplots(nrows=(len(numeric_cols)+2)//3, ncols=3,
                                   figsize=(15, 5*((len(numeric_cols)+2)//3)))
            axes = axes.ravel() if len(numeric_cols) > 1 else [axes]
            
            for idx, col in enumerate(numeric_cols):
                if idx < len(axes):
                    sns.boxplot(data=self.df, x='Churn', y=col, ax=axes[idx])
                    axes[idx].set_title(f'{col} by Churn Status', fontweight='bold')
            
            for idx in range(len(numeric_cols), len(axes)):
                axes[idx].set_visible(False)
                
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/figures/numeric_by_churn_boxplots.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info("Visualizations created successfully")

