import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. Data Loading and Cleaning
# ===============================
df = pd.read_csv('/content/drive/MyDrive/TECH BLITZ 25/TechBlitz DataScience Dataset.csv')
print("Initial Data Shape:", df.shape)
print(df.head())

pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
df_clean = df[(df[pollutant_cols] >= 0).all(axis=1)]
print("Shape after removing bad rows:", df_clean.shape)

# Recode the target: Replace "Poor" with "Bad"
df_clean['Air Quality'] = df_clean['Air Quality'].replace({'Poor': 'Bad'})

# ===============================
# 2. Advanced Feature Engineering
# ===============================
# Create pollution indices using two different weighting schemes
df_clean['Pollution_Index_WHO'] = (0.3 * df_clean['PM2.5'] +
                                   0.2 * df_clean['PM10'] +
                                   0.2 * df_clean['NO2'] +
                                   0.15 * df_clean['SO2'] +
                                   0.15 * df_clean['CO'])
df_clean['Pollution_Index_EPA'] = (0.35 * df_clean['PM2.5'] +
                                   0.15 * df_clean['PM10'] +
                                   0.25 * df_clean['NO2'] +
                                   0.1 * df_clean['SO2'] +
                                   0.15 * df_clean['CO'])

# Industrial risk factor features
df_clean['Industrial_Risk_Factor'] = df_clean['Proximity_to_Industrial_Areas'] * np.log1p(df_clean['Population_Density'])
df_clean['Industrial_Risk_Squared'] = df_clean['Industrial_Risk_Factor'] ** 2
df_clean['Industrial_Population_Interaction'] = np.exp(df_clean['Proximity_to_Industrial_Areas'] / 10) * np.log1p(df_clean['Population_Density'])

# Temperature-Humidity interactions
df_clean['Temp_Humidity_Interaction'] = df_clean['Temperature'] * df_clean['Humidity'] / 100
df_clean['Heat_Index'] = -42.379 + 2.04901523 * df_clean['Temperature'] + 10.14333127 * df_clean['Humidity'] \
                         - 0.22475541 * df_clean['Temperature'] * df_clean['Humidity'] \
                         - 0.00683783 * df_clean['Temperature']**2 - 0.05481717 * df_clean['Humidity']**2 \
                         + 0.00122874 * df_clean['Temperature']**2 * df_clean['Humidity'] \
                         + 0.00085282 * df_clean['Temperature'] * df_clean['Humidity']**2 \
                         - 0.00000199 * df_clean['Temperature']**2 * df_clean['Humidity']**2

# Advanced pollutant ratios
df_clean['PM_Ratio'] = df_clean['PM2.5'] / df_clean['PM10'].replace(0, 0.001)
df_clean['NO2_SO2_Ratio'] = df_clean['NO2'] / df_clean['SO2'].replace(0, 0.001)

# Log and power transformations for pollutants (keeping one set to reduce overhead)
for col in pollutant_cols:
    df_clean[f'Log_{col}'] = np.log1p(df_clean[col])
    df_clean[f'{col}_Squared'] = df_clean[col] ** 2

# Clustering-based feature
kmeans = KMeans(n_clusters=5, random_state=42)
df_clean['Pollutant_Cluster'] = kmeans.fit_predict(df_clean[pollutant_cols])

# Create a couple of combined features for pollution scenarios
df_clean['Industrial_Signature'] = (df_clean['SO2'] * df_clean['NO2']) / (df_clean['PM2.5'] + 0.001)
df_clean['Traffic_Signature'] = (df_clean['NO2'] * df_clean['CO']) / (df_clean['SO2'] + 0.001)

# Fill any NaN values generated from feature engineering
df_clean = df_clean.fillna(0)

# ===============================
# 3. Feature Selection and Processing
# ===============================
# For faster processing, manually define a list of features (keep only engineered features that are likely important)
selected_features = [
    'Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO',
    'Proximity_to_Industrial_Areas', 'Population_Density',
    'Pollution_Index_WHO', 'Pollution_Index_EPA', 'Industrial_Risk_Factor',
    'Temp_Humidity_Interaction', 'Heat_Index', 'PM_Ratio', 'NO2_SO2_Ratio',
    'Log_PM2.5', 'Log_PM10', 'Log_NO2', 'Log_SO2', 'Log_CO',
    'Pollutant_Cluster', 'Industrial_Signature', 'Traffic_Signature'
]

target = 'Air Quality'
le = LabelEncoder()
df_clean[target] = le.fit_transform(df_clean[target])
print("Encoded target classes:", list(le.classes_))

X = df_clean[selected_features]
y = df_clean[target]

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Standardize the feature set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model-based feature selection (using RandomForest)
selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
print(f"Selected {X_train_selected.shape[1]} out of {X_train_scaled.shape[1]} features")

# ===============================
# 4. Hyperparameter Tuning using RandomizedSearchCV
# ===============================
from scipy.stats import randint, uniform

# Define models to try (limiting to faster ones)
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=42)
}

# Simplified hyperparameter grids
param_grids = {
    'RandomForest': {
        'n_estimators': randint(100, 300),
        'max_depth': [15, 20, None],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    },
    'XGBoost': {
        'n_estimators': randint(100, 300),
        'learning_rate': uniform(0.01, 0.1),
        'max_depth': [3, 5, 7]
    },
    'LightGBM': {
        'n_estimators': randint(100, 300),
        'learning_rate': uniform(0.01, 0.1),
        'num_leaves': [31, 63]
    }
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
best_models = {}
best_scores = {}

for model_name, model in models.items():
    print(f"\nTuning hyperparameters for {model_name}...")
    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grids[model_name],
        n_iter=10,            # fewer iterations for speed
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    grid_search.fit(X_train_selected, y_train)
    best_models[model_name] = grid_search.best_estimator_
    best_scores[model_name] = grid_search.best_score_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best CV accuracy for {model_name}: {grid_search.best_score_:.4f}")
    y_pred = grid_search.best_estimator_.predict(X_test_selected)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy for {model_name}: {test_acc:.4f}")

# Determine best model based on cross-validation score
best_model_name = max(best_scores, key=best_scores.get)
best_model = best_models[best_model_name]
print(f"\nBest overall model: {best_model_name} with CV accuracy of {best_scores[best_model_name]:.4f}")

# ===============================
# 5. Ensemble of Top Models
# ===============================
# Using the top 2 models for a fast ensemble
top_models = sorted(best_scores, key=best_scores.get, reverse=True)[:2]
print(f"\nCreating ensemble from top models: {top_models}")
voting_clf = VotingClassifier(
    estimators=[(name, best_models[name]) for name in top_models],
    voting='soft'
)
voting_clf.fit(X_train_selected, y_train)
y_pred_ensemble = voting_clf.predict(X_test_selected)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
ensemble_report = classification_report(y_test, y_pred_ensemble, target_names=le.classes_)
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
print("Ensemble Classification Report:\n", ensemble_report)
