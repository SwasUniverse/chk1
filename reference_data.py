import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

# Ensure directory exists
os.makedirs("model_b", exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for reference data (same as model A)
X, y = make_classification(
    n_samples=1000, 
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    weights=[0.7, 0.3],  # Class imbalance
    random_state=42
)

# Create DataFrame
feature_names = [f'feature_{i+1}' for i in range(5)]
reference_df = pd.DataFrame(X, columns=feature_names)
reference_df['target'] = y

# Add a categorical feature
reference_df['category'] = np.random.choice(['A', 'B', 'C'], size=1000, p=[0.6, 0.3, 0.1])

# Create correlation between feature_1 and target
reference_df['feature_1'] = reference_df['feature_1'] + (reference_df['target'] * 0.5)

# Save reference data
reference_df.to_csv("model_b/reference_data.csv", index=False)
print("Reference data saved to model_b/reference_data.csv")

# Generate production data WITH DATA DRIFT
X_prod, y_prod = make_classification(
    n_samples=800, 
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    weights=[0.7, 0.3],
    random_state=43
)

# Create DataFrame
prod_df = pd.DataFrame(X_prod, columns=feature_names)
prod_df['target'] = y_prod

# Add DATA DRIFT by shifting feature distributions
prod_df['feature_1'] = prod_df['feature_1'] + 1.5  # Shift feature_1 distribution
prod_df['feature_2'] = prod_df['feature_2'] * 1.2  # Scale feature_2 distribution

# Add LABEL DRIFT by changing class distribution
label_drift_mask = np.random.rand(len(prod_df)) < 0.2
prod_df.loc[label_drift_mask, 'target'] = 1 - prod_df.loc[label_drift_mask, 'target']  # Flip 20% of labels

# Add CONCEPT DRIFT by changing feature-target relationship
prod_df['feature_3'] = prod_df['feature_3'] - (prod_df['target'] * 0.7)  # Invert relationship

# Add categorical drift
prod_df['category'] = np.random.choice(['A', 'B', 'C', 'D'], size=800, p=[0.3, 0.3, 0.2, 0.2])  # Different distribution

# Save production data with drift
prod_df.to_csv("model_b/production_data_with_drift.csv", index=False)
print("Production data (with drift) saved to model_b/production_data_with_drift.csv")
