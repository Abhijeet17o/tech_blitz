import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from scipy.stats import randint, uniform
import warnings
warnings.filterwarnings('ignore')

######################################
# 1. DATA LOADING, CLEANING & FEATURE ENGINEERING
######################################
def load_and_engineer_data(file_path):
    df = pd.read_csv(file_path)
    print("Initial Data Shape:", df.shape)
    print(df.head())
    
    pollutant_cols = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO']
    df_clean = df[(df[pollutant_cols] >= 0).all(axis=1)]
    print("Shape after removing bad rows:", df_clean.shape)
    
    # Recode the target: Replace "Poor" with "Bad"
    df_clean['Air Quality'] = df_clean['Air Quality'].replace({'Poor': 'Bad'})
    
    # Create pollution indices using two weighting schemes
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
    
    # Industrial risk features
    df_clean['Industrial_Risk_Factor'] = df_clean['Proximity_to_Industrial_Areas'] * np.log1p(df_clean['Population_Density'])
    df_clean['Industrial_Risk_Squared'] = df_clean['Industrial_Risk_Factor'] ** 2
    df_clean['Industrial_Population_Interaction'] = np.exp(df_clean['Proximity_to_Industrial_Areas'] / 10) * np.log1p(df_clean['Population_Density'])
    
    # Temperature-Humidity interactions
    df_clean['Temp_Humidity_Interaction'] = df_clean['Temperature'] * df_clean['Humidity'] / 100
    df_clean['Heat_Index'] = (-42.379 + 2.04901523 * df_clean['Temperature'] + 10.14333127 * df_clean['Humidity'] -
                              0.22475541 * df_clean['Temperature'] * df_clean['Humidity'] -
                              0.00683783 * df_clean['Temperature']*2 - 0.05481717 * df_clean['Humidity']*2 +
                              0.00122874 * df_clean['Temperature']**2 * df_clean['Humidity'] +
                              0.00085282 * df_clean['Temperature'] * df_clean['Humidity']**2 -
                              0.00000199 * df_clean['Temperature']*2 * df_clean['Humidity']*2)
    
    # Advanced pollutant ratios
    df_clean['PM_Ratio'] = df_clean['PM2.5'] / df_clean['PM10'].replace(0, 0.001)
    df_clean['NO2_SO2_Ratio'] = df_clean['NO2'] / df_clean['SO2'].replace(0, 0.001)
    
    # Log and squared transformations for pollutants
    for col in pollutant_cols:
        df_clean[f'Log_{col}'] = np.log1p(df_clean[col])
        df_clean[f'{col}_Squared'] = df_clean[col] ** 2
        
    # Clustering-based feature
    kmeans = KMeans(n_clusters=5, random_state=42)
    df_clean['Pollutant_Cluster'] = kmeans.fit_predict(df_clean[pollutant_cols])
    
    # Combined features for pollution scenarios
    df_clean['Industrial_Signature'] = (df_clean['SO2'] * df_clean['NO2']) / (df_clean['PM2.5'] + 0.001)
    df_clean['Traffic_Signature'] = (df_clean['NO2'] * df_clean['CO']) / (df_clean['SO2'] + 0.001)
    
    # Fill any NaN values
    df_clean = df_clean.fillna(0)
    
    return df_clean, pollutant_cols

######################################
# 2. MODEL TRAINING & PREDICTION PIPELINE
######################################
def train_models(df_clean, selected_features, target_col):
    # Encode target variable
    le = LabelEncoder()
    df_clean[target_col] = le.fit_transform(df_clean[target_col])
    print("Encoded target classes:", list(le.classes_))
    
    X = df_clean[selected_features]
    y = df_clean[target_col]
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection using RandomForest
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    print(f"Selected {X_train_selected.shape[1]} out of {X_train_scaled.shape[1]} features")
    
    # Define models and hyperparameter grids
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42)
    }
    
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
    model_results = {}  # To store detailed results
    
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
        
        y_pred = grid_search.best_estimator_.predict(X_test_selected)
        test_acc = accuracy_score(y_test, y_pred)
        model_results[model_name] = {
            'best_params': grid_search.best_params_,
            'cv_accuracy': grid_search.best_score_,
            'test_accuracy': test_acc,
            'classification_report': classification_report(y_test, y_pred, target_names=le.classes_)
        }
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best CV accuracy for {model_name}: {grid_search.best_score_:.4f}")
        print(f"Test accuracy for {model_name}: {test_acc:.4f}")
    
    # Determine best overall model based on CV score
    best_model_name = max(best_scores, key=best_scores.get)
    best_model = best_models[best_model_name]
    print(f"\nBest overall model: {best_model_name} with CV accuracy of {best_scores[best_model_name]:.4f}")
    
    # Ensemble using the top 2 models (based on CV score)
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
    
    # Package all outputs into a dictionary
    results = {
        'label_encoder': le,
        'scaler': scaler,
        'selector': selector,
        'X_train_selected': X_train_selected,
        'X_test_selected': X_test_selected,
        'best_models': best_models,
        'model_results': model_results,
        'best_overall_model': (best_model_name, best_model),
        'ensemble': {
            'classifier': voting_clf,
            'accuracy': ensemble_accuracy,
            'classification_report': ensemble_report
        }
    }
    
    return results

######################################
# 3. VISUALIZATION FUNCTIONS FOR DASHBOARD
######################################
def create_class_distribution_plot(df):
    class_counts = df['Air Quality'].value_counts().reset_index()
    class_counts.columns = ['Air Quality', 'Count']
    fig = px.bar(
        class_counts, 
        x='Air Quality', 
        y='Count',
        color='Air Quality',
        title='Distribution of Air Quality Classes',
        text='Count'
    )
    fig.update_layout(xaxis_title='Air Quality Class', yaxis_title='Count', showlegend=False)
    return fig

def create_pollutant_boxplots(df, pollutant_cols):
    """Create interactive boxplots of pollutants grouped by air quality."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # We will have one boxplot per pollutant in a 3x2 grid,
    # plus one for 'Pollution_Index_WHO' in the last subplot.
    fig = make_subplots(
        rows=3, 
        cols=2,
        subplot_titles=[f'{col} by Air Quality' for col in pollutant_cols] 
                       + ['Pollution Index (WHO) by Air Quality']
    )
    
    # Define color mapping for each air quality category
    color_map = {'Good': '#66c2a5', 'Moderate': '#fc8d62', 'Bad': '#8da0cb'}
    
    # Plot each pollutant in the first 5 subplots
    for i, col in enumerate(pollutant_cols):
        row = i // 2 + 1
        col_pos = i % 2 + 1
        
        for quality, color in color_map.items():
            subset = df[df['Air Quality'] == quality]
            box = go.Box(
                y=subset[col],
                name=quality,
                marker_color=color,
                showlegend=(i == 0),  # Only show legend for the first pollutant
                legendgroup=quality
            )
            fig.add_trace(box, row=row, col=col_pos)
        
        # Give each subplot a y-axis title matching the pollutant
        fig.update_yaxes(title_text=col, row=row, col=col_pos)
    
    # In the last subplot (3,2), show boxplots for Pollution_Index_WHO
    for quality, color in color_map.items():
        subset = df[df['Air Quality'] == quality]
        box = go.Box(
            y=subset['Pollution_Index_WHO'],
            name=quality,
            marker_color=color,
            showlegend=False,  # Already shown in the first pollutant's legend
            legendgroup=quality
        )
        fig.add_trace(box, row=3, col=2)
    
    # Label the y-axis for the last subplot
    fig.update_yaxes(title_text="Pollution_Index_WHO", row=3, col=2)
    
    # Configure the overall layout
    fig.update_layout(
        title_text='Pollutant Levels by Air Quality Class',
        legend_title_text='Air Quality',
        boxmode='group',            # Group the boxplots side-by-side
        height=1000, 
        width=1200,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    return fig


def create_correlation_heatmap(df, pollutant_cols):
    core_features = pollutant_cols + ['Temperature', 'Humidity', 
                                        'Proximity_to_Industrial_Areas', 
                                        'Population_Density', 'Pollution_Index_WHO']
    corr_matrix = df[core_features].corr().round(2)
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Correlation Heatmap of Key Features',
        labels=dict(color="Correlation")
    )
    fig.update_layout(height=700, width=800)
    return fig

def create_pca_visualization(df, pollutant_cols):
    features = pollutant_cols + ['Temperature', 'Humidity', 
                                 'Proximity_to_Industrial_Areas', 
                                 'Population_Density']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
    pca_df['Air Quality'] = df['Air Quality'].values
    fig = px.scatter(
        pca_df,
        x='PC1',
        y='PC2',
        color='Air Quality',
        title=f'PCA Projection of Air Quality Data<br>Explained Variance: {sum(pca.explained_variance_ratio_):.2%}',
        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=10))
    return fig

def create_pollution_temperature_scatter(df):
    fig = px.scatter(
        df,
        x='Temperature',
        y='Pollution_Index_WHO',
        size='Humidity',
        color='Air Quality',
        hover_name='Air Quality',
        hover_data=["PM2.5", "PM10", "NO2", "SO2", "CO"],
        title='Pollution Index vs. Temperature',
        labels={'Temperature': 'Temperature', 'Pollution_Index_WHO': 'Pollution Index (WHO)', 'Humidity': 'Humidity'},
        size_max=30
    )
    fig.update_layout(title_font_size=18, xaxis_title='Temperature', yaxis_title='Pollution Index (WHO)')
    return fig

def create_industrial_impact_plot(df):
    df_with_bins = df.copy()
    df_with_bins['Proximity_Category'] = pd.cut(
        df['Proximity_to_Industrial_Areas'], 
        bins=5, 
        labels=['Very Far', 'Far', 'Moderate', 'Close', 'Very Close']
    )
    grouped_data = df_with_bins.groupby(['Proximity_Category', 'Air Quality'])['Pollution_Index_WHO'].mean().reset_index()
    fig = px.bar(
        grouped_data,
        x='Proximity_Category',
        y='Pollution_Index_WHO',
        color='Air Quality',
        barmode='group',
        title='Pollution Index by Industrial Proximity and Air Quality',
        labels={'Proximity_Category': 'Proximity to Industrial Areas', 'Pollution_Index_WHO': 'Avg Pollution Index (WHO)'}
    )
    fig.update_xaxes(categoryorder='array', categoryarray=['Very Far', 'Far', 'Moderate', 'Close', 'Very Close'])
    return fig

######################################
# 4. DASHBOARD CREATION: COMBINING VISUALS AND MODEL OUTPUTS
######################################
def create_dashboard(file_path='TechBlitz DataScience Dataset.csv', dashboard_filename='air_quality_dashboard.html'):
    # Load and process data
    df_clean, pollutant_cols = load_and_engineer_data(file_path)
    
    # Define selected features for modeling (keep key engineered ones)
    selected_features = [
        'Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO',
        'Proximity_to_Industrial_Areas', 'Population_Density',
        'Pollution_Index_WHO', 'Pollution_Index_EPA', 'Industrial_Risk_Factor',
        'Temp_Humidity_Interaction', 'Heat_Index', 'PM_Ratio', 'NO2_SO2_Ratio',
        'Log_PM2.5', 'Log_PM10', 'Log_NO2', 'Log_SO2', 'Log_CO',
        'Pollutant_Cluster', 'Industrial_Signature', 'Traffic_Signature'
    ]
    target = 'Air Quality'
    
    # Train models and get prediction outputs
    results = train_models(df_clean, selected_features, target)
    
    # Prepare textual output for model performance
    model_text = "<h2>Model Performance & Prediction Metrics</h2>"
    for model_name, res in results['model_results'].items():
        model_text += f"<h3>{model_name}</h3>"
        model_text += f"<p><strong>Best Parameters:</strong> {res['best_params']}</p>"
        model_text += f"<p><strong>CV Accuracy:</strong> {res['cv_accuracy']:.4f}</p>"
        model_text += f"<p><strong>Test Accuracy:</strong> {res['test_accuracy']:.4f}</p>"
        model_text += f"<pre>{res['classification_report']}</pre>"
    
    # Ensemble performance
    ensemble = results['ensemble']
    model_text += f"<h3>Ensemble Model (Voting Classifier)</h3>"
    model_text += f"<p><strong>Ensemble Test Accuracy:</strong> {ensemble['accuracy']:.4f}</p>"
    model_text += f"<pre>{ensemble['classification_report']}</pre>"
    
    # Create interactive visualizations
    class_dist_fig = create_class_distribution_plot(df_clean)
    pollutant_box_fig = create_pollutant_boxplots(df_clean, pollutant_cols)
    corr_fig = create_correlation_heatmap(df_clean, pollutant_cols)
    pca_fig = create_pca_visualization(df_clean, pollutant_cols)
    scatter_fig = create_pollution_temperature_scatter(df_clean)
    industrial_fig = create_industrial_impact_plot(df_clean)
    
    # Write everything into a single HTML file
    with open(dashboard_filename, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Air Quality Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                .chart-container {{
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .row {{
                    display: flex;
                    flex-wrap: wrap;
                    margin: -10px;
                }}
                .col {{
                    flex: 1;
                    min-width: 400px;
                    padding: 10px;
                }}
                @media (max-width: 900px) {{
                    .col {{
                        flex: 100%;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Air Quality Analysis Dashboard</h1>
                <div class="chart-container">
                    {model_text}
                </div>
                <div class="chart-container">
                    <div id="class_dist"></div>
                </div>
                <div class="chart-container">
                    <div id="pollutant_box"></div>
                </div>
                <div class="row">
                    <div class="col chart-container">
                        <div id="correlation"></div>
                    </div>
                    <div class="col chart-container">
                        <div id="pca"></div>
                    </div>
                </div>
                <div class="row">
                    <div class="col chart-container">
                        <div id="scatter"></div>
                    </div>
                    <div class="col chart-container">
                        <div id="industrial"></div>
                    </div>
                </div>
            </div>
            <script>
                var class_dist = {class_dist_fig.to_json()};
                Plotly.newPlot("class_dist", class_dist.data, class_dist.layout);
                
                var pollutant_box = {pollutant_box_fig.to_json()};
                Plotly.newPlot("pollutant_box", pollutant_box.data, pollutant_box.layout);
                
                var correlation = {corr_fig.to_json()};
                Plotly.newPlot("correlation", correlation.data, correlation.layout);
                
                var pca = {pca_fig.to_json()};
                Plotly.newPlot("pca", pca.data, pca.layout);
                
                var scatter = {scatter_fig.to_json()};
                Plotly.newPlot("scatter", scatter.data, scatter.layout);
                
                var industrial = {industrial_fig.to_json()};
                Plotly.newPlot("industrial", industrial.data, industrial.layout);
                
                // Optional: add window resize listener for responsiveness
                window.onresize = function() {{
                    Plotly.Plots.resize(document.getElementById('class_dist'));
                    Plotly.Plots.resize(document.getElementById('pollutant_box'));
                    Plotly.Plots.resize(document.getElementById('correlation'));
                    Plotly.Plots.resize(document.getElementById('pca'));
                    Plotly.Plots.resize(document.getElementById('scatter'));
                    Plotly.Plots.resize(document.getElementById('industrial'));
                }};
            </script>
        </body>
        </html>
        """)
    
    print(f"Dashboard created and saved as '{dashboard_filename}'. Open this file in a web browser to view the interactive visualizations and model predictions.")

######################################
# 5. MAIN
######################################
if _name_ == "_main_":
    # Provide the path to your dataset
    data_file_path = 'TechBlitz DataScience Dataset.csv'
    create_dashboard(file_path=data_file_path)
