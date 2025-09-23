import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Load the dataset
df = pd.read_csv('augment1.csv')

# Identify string columns that need to be excluded or encoded
string_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"String columns that will be excluded: {string_columns}")

# Create a simple dataset with only numeric columns for the model
df_numeric = df.select_dtypes(include=['int64', 'float64'])
print(f"Numeric columns that will be used: {df_numeric.columns.tolist()}")

# Split into features and target
X = df_numeric.drop('tg', axis=1)
y = df_numeric['tg']

# Train a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

# Sort by importance
feature_importances = feature_importances.sort_values('Importance', ascending=False).reset_index(drop=True)

# Display top features
print("\nTop features by importance:")
print(feature_importances)

# Visualize feature importances
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Features by Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as 'feature_importance.png'")