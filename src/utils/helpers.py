import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Union
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def save_object(obj: Any, filepath: str) -> None:   # save object to file using joblib

    dir_path = Path(filepath).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)


def load_object(filepath: str) -> Any:              # load object from file using joblib
    return joblib.load(filepath)


def save_json(data: Dict, filepath: str) -> None:  # save dictionary as JSON file

    dir_path = Path(filepath).parent
    dir_path.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath: str) -> Dict:               # load dictionary from JSON file
    with open(filepath, 'r') as f:
        return json.load(f)
    

def create_directory_structure():           # create standard directory structure for project
    
    directories = [
        "data/raw", "data/processed", "data/features", "data/external",
        "notebooks", "src/data", "src/models", "src/evaluation", 
        "src/optimization", "src/deployment", "src/utils",
        "tests/test_data", "tests/test_models", "tests/test_evaluation",
        "configs", "scripts", "docs", "reports/figures",
        "models/trained_models", "models/checkpoints", "models/experiment_logs",
        "deployments/docker", "deployments/kubernetes", "deployments/monitoring",
        "environment", "logs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

        if directory.startswith("src/"):
            init_file = Path(directory) / "__init__.py"
            init_file.touch()

        print(" Directory structure created successfully!")

def calculate_business_metrics(y_true: np.ndarray, y_pred: np.nadarray,
                               monthly_charges: np.ndarray = None) -> Dict[str, Any]:   # calculate business specific metrics for churn prediction
    
    from sklearn.metrics import confusion_matrix

    y_true_binary = (y_true == 1) if isinstance(y_true[0], (int, float)) else (y_true == 'Yes')
    y_pred_binary = (y_pred == 1) if isinstance(y_pred[0], (int, float)) else (y_pred == 'Yes')

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

    metrics = {
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'customers_identified_correctly': tp,
        'customers_missed': fn,
        'false_alarms': fp
    }

    if monthly_charges is not None:

        actual_churners_mask = y_true_binary
        revenue_at_risk = monthly_charges[actual_churners_mask].sum() 

        correctly_identified_mask = y_true_binary & y_pred_binary
        revenue_potentially_saved = monthly_charges[correctly_identified_mask].sum()

        false_alarm_mask = (~y_true_binary) & y_pred_binary
        false_alarm_cost = monthly_charges[false_alarm_mask].sum()

        metrics.update({
            'total_revenue_at_risk': revenue_at_risk,
            'revenue_potentially_saved': revenue_potentially_saved,
            'false_alarm_cost': false_alarm_cost,
            'revenue_capture_rate': revenue_potentially_saved / revenue_at_risk if revenue_at_risk > 0 else 0
        })

    return metrics

def plot_feature_importance(feature_names: List[str], importances: np.ndarray,      # plot feature importance
                            title: str = "Feature Importance", top_n: int = 20,
                            figsize: Tuple[int, int] = (10, 8)) -> None:
    
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=figsize)
    plt.barh(range(len(top_features)), top_importances[::-1])
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def create_model_comparison_plot(results: Dict[str, Dict[str, float]],          # create model comparison plot
                                 metric: str = 'roc_auc',
                                 figsize: Tuple[int, int] = (12, 6)) -> None:
    
    models = list(results.keys())
    scores = [results[model].get(metric, 0) for model in models]

    plt.figure(figsize=figsize)
    bars = plt.bar(models, scores, alpha=0.7, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel(f'{metric.upper()}')
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xticks(rotation=45, ha='right')

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.show()







    
