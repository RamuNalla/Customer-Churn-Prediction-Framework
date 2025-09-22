
import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_telco_data():                     # Create sample Telco Customer Churn dataset

    print(" Creating sample Telco Customer Churn dataset...")
    
    data_dir = Path("data/raw")                     # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    n_samples = 7043
     
    data = {                                        # Generate sample data
        'customerID': [f'ID_{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.52, 0.48]),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
        'tenure': np.random.randint(1, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.90, 0.10]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.42, 0.48, 0.10]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.34, 0.44, 0.22]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.28, 0.50, 0.22]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.34, 0.44, 0.22]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.29, 0.49, 0.22]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.38, 0.40, 0.22]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.38, 0.40, 0.22]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.21, 0.24]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.59, 0.41]),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_samples, p=[0.33, 0.23, 0.22, 0.22]),
        'MonthlyCharges': np.round(np.random.uniform(18.25, 118.75, n_samples), 2),
        'TotalCharges': np.round(np.random.uniform(18.8, 8684.8, n_samples), 2)
    }
    
    churn_prob = np.random.random(n_samples)                    # Create realistic churn based on business logic
    churn_factors = np.zeros(n_samples)
    churn_factors += (np.array(data['Contract']) == 'Month-to-month') * 0.3
    churn_factors += (np.array(data['MonthlyCharges']) > 80) * 0.2
    churn_factors += (np.array(data['tenure']) < 12) * 0.3
    churn_factors += (np.array(data['InternetService']) == 'Fiber optic') * 0.15
    churn_factors += (np.array(data['PaymentMethod']) == 'Electronic check') * 0.1
    
    base_churn_rate = 0.2
    final_churn_prob = base_churn_rate + churn_factors
    data['Churn'] = np.where(churn_prob < final_churn_prob, 'Yes', 'No')
    
    df = pd.DataFrame(data)                                     # Create DataFrame and introduce some missing values
    missing_indices = np.random.choice(df.index, size=int(0.001 * len(df)), replace=False)
    df.loc[missing_indices, 'TotalCharges'] = ''
    
    output_path = data_dir / "telco_customer_churn.csv"         # Save dataset
    df.to_csv(output_path, index=False)
    
    print(f" Dataset created: {output_path}")
    print(f" Dataset shape: {df.shape}")
    print(f" Churn rate: {(df['Churn'] == 'Yes').mean()*100:.1f}%")
    
    # Create data dictionary
    data_dict = """
Telco Customer Churn Dataset - Data Dictionary
==============================================

customerID: Customer ID (string)
gender: Customer gender (Male/Female)  
SeniorCitizen: Whether customer is senior citizen (0/1)
Partner: Whether customer has partner (Yes/No)
Dependents: Whether customer has dependents (Yes/No)
tenure: Number of months customer has stayed (numeric)
PhoneService: Whether customer has phone service (Yes/No)
MultipleLines: Whether customer has multiple lines (Yes/No/No phone service)
InternetService: Customer's internet service provider (DSL/Fiber optic/No)
OnlineSecurity: Whether customer has online security (Yes/No/No internet service)
OnlineBackup: Whether customer has online backup (Yes/No/No internet service)
DeviceProtection: Whether customer has device protection (Yes/No/No internet service)
TechSupport: Whether customer has tech support (Yes/No/No internet service)
StreamingTV: Whether customer has streaming TV (Yes/No/No internet service)
StreamingMovies: Whether customer has streaming movies (Yes/No/No internet service)
Contract: Customer contract term (Month-to-month/One year/Two year)
PaperlessBilling: Whether customer has paperless billing (Yes/No)
PaymentMethod: Customer's payment method (Electronic check/Mailed check/Bank transfer/Credit card)
MonthlyCharges: Monthly charge amount (numeric)
TotalCharges: Total charges (numeric, may have missing values)
Churn: Whether customer churned (Yes/No) - TARGET VARIABLE
    """
    
    with open(data_dir / "data_dictionary.txt", 'w') as f:
        f.write(data_dict)
    
    print(" Data dictionary created")
    return output_path

def main():
    create_sample_telco_data()

if __name__ == "__main__":
    main()