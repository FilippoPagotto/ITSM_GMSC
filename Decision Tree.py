import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
csv_path = r'E:\PyCharm\pythonProject4\dft_traffic_counts_raw_counts3.csv'
data = pd.read_csv(csv_path)

# Handle missing values
data.fillna(data.mean(), inplace=True)

# Label Encoding for categorical columns
label_encoder = LabelEncoder()
for column in ['Direction_of_travel', 'Region_id', 'Local_authority_id', 'Road_category', 'Road_type',
               'Road_name', 'Start_junction_road_name', 'End_junction_road_name']:
    data[column] = label_encoder.fit_transform(data[column])

# Extract features and target variable
X = data.drop(['All_motor_vehicles'], axis=1)  # Features (excluding the target variable)
y = data['All_motor_vehicles']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using correlation matrix
correlation_matrix = X_train.corr()
selected_features_corr = SelectKBest(f_regression, k=8).fit(X_train, y_train).get_support()
selected_columns_corr = X_train.columns[selected_features_corr]
X_train_selected_corr = X_train[selected_columns_corr]
X_test_selected_corr = X_test[selected_columns_corr]

# Feature selection using ANOVA
selected_features_anova = SelectKBest(f_classif, k=8).fit(X_train, y_train).get_support()
selected_columns_anova = X_train.columns[selected_features_anova]
X_train_selected_anova = X_train[selected_columns_anova]
X_test_selected_anova = X_test[selected_columns_anova]

# Create and train the Decision Tree model using correlation-based features
max_depth_corr = 20  # You can adjust this value
model_corr = DecisionTreeRegressor(max_depth=max_depth_corr)
model_corr.fit(X_train_selected_corr, y_train)

# Create and train the Decision Tree model using ANOVA-based features
max_depth_anova = 20  # You can adjust this value
model_anova = DecisionTreeRegressor(max_depth=max_depth_anova)
model_anova.fit(X_train_selected_anova, y_train)

# Make predictions on the test set for correlation-based features
y_pred_corr = model_corr.predict(X_test_selected_corr)

# Make predictions on the test set for ANOVA-based features
y_pred_anova = model_anova.predict(X_test_selected_anova)

# Calculate RMSE and R2 for correlation-based features
rmse_corr = np.sqrt(mean_squared_error(y_test, y_pred_corr))
r2_corr = r2_score(y_test, y_pred_corr)

# Calculate RMSE and R2 for ANOVA-based features
rmse_anova = np.sqrt(mean_squared_error(y_test, y_pred_anova))
r2_anova = r2_score(y_test, y_pred_anova)

# Display results for correlation-based features
print("Results for Correlation-based Feature Selection with Decision Tree:")
print(f"Selected Features: {selected_columns_corr}")
print(f"Root Mean Squared Error (RMSE): {rmse_corr}")
print(f"R-squared (R2): {r2_corr}")

# Display results for ANOVA-based features
print("\nResults for ANOVA-based Feature Selection with Decision Tree:")
print(f"Selected Features: {selected_columns_anova}")
print(f"Root Mean Squared Error (RMSE): {rmse_anova}")
print(f"R-squared (R2): {r2_anova}")

# Heatmap of the selected features for the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix.loc[selected_columns_corr, selected_columns_corr], annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Selected Features (Correlation-based)")
plt.show()

# Heatmap of the selected features for ANOVA
plt.figure(figsize=(12, 10))
sns.heatmap(X_train_selected_anova.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Selected Features (ANOVA-based)")
plt.show()

# Scatter plot comparing predicted vs actual values with lines for correlation-based features
step = 10  # Set the step variable
n_corr = len(y_test) - len(y_test) % step
plt.figure(figsize=(12, 8))
plt.plot(np.linspace(0, 100, n_corr//step), y_test[:n_corr:step], color='blue', label='Actual Data', linestyle='-', linewidth=1, alpha=0.5)
plt.plot(np.linspace(0, 100, n_corr//step), y_pred_corr[:n_corr:step], color='red', label='Predicted Data (Correlation-based)', linestyle='-', linewidth=1, alpha=0.5)
plt.xlabel("Index")
plt.ylabel("All_motor_vehicles")
plt.title(f"Comparison of Actual vs Predicted Values (Every {step}th Index) - Correlation-based")
plt.legend()
plt.show()

# Scatter plot comparing predicted vs actual values with lines for ANOVA-based features
n_anova = len(y_test) - len(y_test) % step
plt.figure(figsize=(12, 8))
plt.plot(np.linspace(0, 100, n_anova//step), y_test[:n_anova:step], color='blue', label='Actual Data', linestyle='-', linewidth=1, alpha=0.5)
plt.plot(np.linspace(0, 100, n_anova//step), y_pred_anova[:n_anova:step], color='green', label='Predicted Data (ANOVA-based)', linestyle='-', linewidth=1, alpha=0.5)
plt.xlabel("Index")
plt.ylabel("All_motor_vehicles")
plt.title(f"Comparison of Actual vs Predicted Values (Every {step}th Index) - ANOVA-based")
plt.legend()
plt.show()
