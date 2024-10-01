import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/vishnu/2_docking - tryosine_kinase_data_docking.csv'  
data = pd.read_csv(file_path)

# Preserve the 'Title' column as ligand identifiers
ligand_identifiers = data['Title'].copy()

# Preprocessing
data = data.dropna(subset=['docking score'])
threshold = 0.5 * len(data)
data = data.dropna(thresh=threshold, axis=1)
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())
categorical_columns = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Splitting the data
features = data.drop('docking score', axis=1)
target = data['docking score']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Preserving the 'Title' column for the test set
X_test_with_title = X_test.copy()
X_test_with_title['Title'] = ligand_identifiers.iloc[X_test.index]

# Training the Random Forest model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predicting the docking scores
y_pred_rf = rf_model.predict(X_test)

# Creating a dataframe for actual vs. predicted scores
ligand_scores = pd.DataFrame({
    'Ligand Identifier': X_test_with_title['Title'],
    'Actual Docking Score': y_test,
    'Predicted Docking Score': y_pred_rf
}).reset_index(drop=True)


# Display the actual vs. predicted scores for specific ligands
print("Actual vs. Predicted Docking Scores for Selected Ligands:")
selected_ligands = ['AQ4 - Erlotinib', 'IRE - gefitinib', '0WN']
selected_scores = ligand_scores[ligand_scores['Ligand Identifier'].isin(selected_ligands)]
print(selected_scores)

# Feature importance plot
feature_importances = rf_model.feature_importances_
feature_names = X_train.columns
importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importances_df.head(10))
plt.title('Top 10 Feature Importances in Random Forest Model')
plt.show()

# Actual vs. Predicted plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Docking Scores')
plt.xlabel('Actual Docking Score')
plt.ylabel('Predicted Docking Score')
plt.show()



# Load the new ligand data
new_ligand_path = '/Users/vishnu/zd6.csv'
new_ligand_data = pd.read_csv(new_ligand_path)

# Preprocess the new ligand data
# Fill missing values in numerical columns with the mean from the training set
numerical_columns_new_ligand = [col for col in X_train.select_dtypes(include=['float64', 'int64']).columns]
new_ligand_data[numerical_columns_new_ligand] = new_ligand_data[numerical_columns_new_ligand].fillna(X_train[numerical_columns_new_ligand].mean())

# Add missing dummy columns that were in the training data but not in the new ligand data
for col in X_train.columns:
    if col not in new_ligand_data.columns:
        new_ligand_data[col] = 0

# Reorder columns to match the training data
new_ligand_data = new_ligand_data[X_train.columns]

# Predict the docking score for the new ligand
predicted_docking_score_new_ligand = rf_model.predict(new_ligand_data)

# Print the predicted docking score
print("Predicted Docking Score for the new ligand (ZD6):", predicted_docking_score_new_ligand[0])
