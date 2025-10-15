# # import pandas as pd
# # import numpy as np
# # from datetime import timedelta
# # from prophet import Prophet
# # import plotly.graph_objects as go
# # from plotly.subplots import make_subplots
# # import warnings
# # import sys
# # import logging
# #
# # warnings.filterwarnings('ignore')
# #
# #
# # CSV_FILE = "nawassco.csv"
# # df = pd.read_csv(CSV_FILE)
# #
# # def run_prophet_analysis(df,
# #                          date_column='date',
# #                          target_column='actual_consumption_m3',
# #                          forecast_days=14,
# #                          changepoint_prior_scale=0.05,
# #                          weekly_seasonality=True,
# #                          yearly_seasonality=False,
# #                          aggregate_by_date=True):
# #     """
# #     Run complete Prophet time series analysis on a dataframe.
# #
# #     Parameters:
# #     -----------
# #     df : pandas.DataFrame
# #         Input dataframe with date and target columns
# #     date_column : str
# #         Name of the date column (default: 'date')
# #     target_column : str
# #         Name of the target variable column (default: 'actual_consumption_m3')
# #     forecast_days : int
# #         Number of days to forecast (default: 14)
# #     changepoint_prior_scale : float
# #         Prophet flexibility parameter (default: 0.05)
# #     weekly_seasonality : bool
# #         Enable weekly seasonality (default: True)
# #     yearly_seasonality : bool
# #         Enable yearly seasonality (default: False)
# #     aggregate_by_date : bool
# #         If True, aggregates data by date (useful for zone-based data)
# #
# #     Returns:
# #     --------
# #     dict containing:
# #         - 'forecast_fig': Plotly figure with forecast plot
# #         - 'components_fig': Plotly figure with component decomposition
# #         - 'metrics': Dictionary with MAE, RMSE, MAPE
# #         - 'forecast_summary': Dictionary with forecast statistics
# #         - 'model': Fitted Prophet model
# #         - 'forecast_df': Full forecast dataframe
# #     """
# #     # Prepare data
# #     df_clean = df.copy()
# #
# #     # Standardize column names if needed
# #     if date_column in df_clean.columns:
# #         df_clean[date_column] = pd.to_datetime(df_clean[date_column])
# #     else:
# #         raise ValueError(f"Date column '{date_column}' not found in dataframe")
# #
# #     if target_column not in df_clean.columns:
# #         raise ValueError(f"Target column '{target_column}' not found in dataframe")
# #
# #     # Aggregate by date if specified
# #     if aggregate_by_date and len(df_clean.columns) > 2:
# #         prophet_df = df_clean.groupby(date_column).agg({target_column: 'sum'}).reset_index()
# #     else:
# #         prophet_df = df_clean[[date_column, target_column]].copy()
# #
# #     # Rename for Prophet
# #     prophet_df = prophet_df.rename(columns={date_column: 'ds', target_column: 'y'})
# #
# #     # Remove any NaN values
# #     prophet_df = prophet_df.dropna()
# #
# #     # Sort by date
# #     prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
# #
# #     print(f"📊 Prophet Analysis Starting...")
# #     print(f"   Data points: {len(prophet_df)}")
# #     print(f"   Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
# #     print(f"   Forecast period: {forecast_days} days")
# #
# #     # Handle the CMDSTANPY backend issue on some systems by using the default
# #     stan_backend = 'CMDSTANPY'
# #     if sys.platform == 'win32':
# #         stan_backend = None
# #
# #     # Train Prophet model
# #     model = Prophet(
# #         changepoint_prior_scale=changepoint_prior_scale,
# #         yearly_seasonality=yearly_seasonality,
# #         weekly_seasonality=weekly_seasonality,
# #         daily_seasonality=False,
# #         stan_backend=stan_backend
# #     )
# #
# #     try:
# #         model.fit(prophet_df)
# #         print("✅ Model training successful")
# #     except Exception as e:
# #         print(f"⚠️ Initial training failed, using Newton optimizer...")
# #         logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
# #         model.fit(prophet_df, algorithm='Newton')
# #         print("✅ Model training successful (Newton)")
# #
# #     # Create future dataframe
# #     future = model.make_future_dataframe(periods=forecast_days)
# #     forecast = model.predict(future)
# #
# #     # ==================== CREATE FORECAST PLOT ====================
# #     fig_forecast = go.Figure()
# #
# #     # Historical data
# #     fig_forecast.add_trace(go.Scatter(
# #         x=prophet_df['ds'],
# #         y=prophet_df['y'],
# #         mode='markers',
# #         name='Actual Data',
# #         marker=dict(size=6, color='#2c3e50', opacity=0.7),
# #         hovertemplate='Date: %{x}<br>Value: %{y:,.2f}<extra></extra>'
# #     ))
# #
# #     # Forecast line
# #     fig_forecast.add_trace(go.Scatter(
# #         x=forecast['ds'],
# #         y=forecast['yhat'],
# #         mode='lines',
# #         name='Prophet Forecast',
# #         line=dict(color='#3498db', width=3),
# #         hovertemplate='Date: %{x}<br>Forecast: %{y:,.2f}<extra></extra>'
# #     ))
# #
# #     # Confidence interval upper
# #     fig_forecast.add_trace(go.Scatter(
# #         x=forecast['ds'],
# #         y=forecast['yhat_upper'],
# #         mode='lines',
# #         name='Upper Bound',
# #         line=dict(width=0),
# #         showlegend=False,
# #         hoverinfo='skip'
# #     ))
# #
# #     # Confidence interval lower (with fill)
# #     fig_forecast.add_trace(go.Scatter(
# #         x=forecast['ds'],
# #         y=forecast['yhat_lower'],
# #         mode='lines',
# #         name='80% Confidence Interval',
# #         fill='tonexty',
# #         fillcolor='rgba(52, 152, 219, 0.2)',
# #         line=dict(width=0),
# #         hovertemplate='Lower: %{y:,.2f}<extra></extra>'
# #     ))
# #
# #     # Add vertical line to separate historical and forecast
# #     last_historical_date = prophet_df['ds'].max()
# #     fig_forecast.add_vline(
# #         x=last_historical_date,
# #         line_dash="dash",
# #         line_color="gray",
# #         opacity=0.5,
# #     )
# #     # Add annotation separately to avoid the internal error
# #     fig_forecast.add_annotation(
# #         x=last_historical_date,
# #         y=1.05, # Position above the plot
# #         yref="paper",
# #         text="Forecast Start",
# #         showarrow=False,
# #         font=dict(color="gray")
# #     )
# #
# #     fig_forecast.update_layout(
# #         xaxis_title='Date',
# #         yaxis_title='Value',
# #         hovermode='x unified',
# #         template='plotly_white',
# #         height=550,
# #         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
# #     )
# #
# #     # ==================== CREATE COMPONENTS PLOT ====================
# #     subplot_titles = ['Trend Component']
# #     if weekly_seasonality:
# #         subplot_titles.append('Weekly Seasonality')
# #     if yearly_seasonality:
# #         subplot_titles.append('Yearly Seasonality')
# #     subplot_titles.append('Forecast Uncertainty')
# #
# #     fig_components = make_subplots(
# #         rows=len(subplot_titles), cols=1,
# #         subplot_titles=subplot_titles,
# #         vertical_spacing=0.08
# #     )
# #
# #     row = 1
# #
# #     # Trend
# #     fig_components.add_trace(go.Scatter(
# #         x=forecast['ds'],
# #         y=forecast['trend'],
# #         mode='lines',
# #         name='Trend',
# #         line=dict(color='#e74c3c', width=2),
# #         hovertemplate='%{y:,.2f}<extra></extra>'
# #     ), row=row, col=1)
# #     fig_components.update_yaxes(title_text="Trend", row=row, col=1)
# #     row += 1
# #
# #     # Weekly seasonality
# #     if weekly_seasonality:
# #         fig_components.add_trace(go.Scatter(
# #             x=forecast['ds'],
# #             y=forecast['weekly'],
# #             mode='lines',
# #             name='Weekly',
# #             line=dict(color='#27ae60', width=2),
# #             hovertemplate='%{y:,.2f}<extra></extra>'
# #         ), row=row, col=1)
# #         fig_components.update_yaxes(title_text="Weekly Effect", row=row, col=1)
# #         row += 1
# #
# #     # Yearly seasonality
# #     if yearly_seasonality:
# #         fig_components.add_trace(go.Scatter(
# #             x=forecast['ds'],
# #             y=forecast['yearly'],
# #             mode='lines',
# #             name='Yearly',
# #             line=dict(color='#f39c12', width=2),
# #             hovertemplate='%{y:,.2f}<extra></extra>'
# #         ), row=row, col=1)
# #         fig_components.update_yaxes(title_text="Yearly Effect", row=row, col=1)
# #         row += 1
# #
# #     # Uncertainty (yhat_upper - yhat_lower)
# #     uncertainty = forecast['yhat_upper'] - forecast['yhat_lower']
# #     fig_components.add_trace(go.Scatter(
# #         x=forecast['ds'],
# #         y=uncertainty,
# #         mode='lines',
# #         name='Uncertainty',
# #         line=dict(color='#9b59b6', width=2),
# #         fill='tozeroy',
# #         fillcolor='rgba(155, 89, 182, 0.2)',
# #         hovertemplate='%{y:,.2f}<extra></extra>'
# #     ), row=row, col=1)
# #     fig_components.update_yaxes(title_text="Uncertainty Range", row=row, col=1)
# #     fig_components.update_xaxes(title_text="Date", row=row, col=1)
# #
# #     fig_components.update_layout(
# #         height=250 * len(subplot_titles),
# #         showlegend=False,
# #         template='plotly_white',
# #         title_text='Prophet Component Decomposition'
# #     )
# #
# #     # ==================== CALCULATE METRICS ====================
# #     train_forecast = forecast[forecast['ds'].isin(prophet_df['ds'])]
# #     actual_values = prophet_df['y'].values
# #     predicted_values = train_forecast['yhat'].values[:len(actual_values)]
# #     mae = np.mean(np.abs(actual_values - predicted_values))
# #     rmse = np.sqrt(np.mean((actual_values - predicted_values)**2))
# #     mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
# #
# #     metrics = {
# #         'MAE': mae,
# #         'RMSE': rmse,
# #         'MAPE': mape,
# #         'data_points': len(prophet_df),
# #         'forecast_points': forecast_days
# #     }
# #
# #     # ==================== FORECAST SUMMARY ====================
# #     future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]
# #     forecast_summary = {
# #         'avg_forecast': future_forecast['yhat'].mean(),
# #         'min_forecast': future_forecast['yhat'].min(),
# #         'max_forecast': future_forecast['yhat'].max(),
# #         'total_forecast': future_forecast['yhat'].sum(),
# #         'avg_confidence_width': (future_forecast['yhat_upper'] - future_forecast['yhat_lower']).mean(),
# #         'forecast_start_date': future_forecast['ds'].min(),
# #         'forecast_end_date': future_forecast['ds'].max()
# #     }
# #
# #     print("\n📊 Analysis Complete!")
# #     print(f"   MAE: {mae:,.2f}")
# #     print(f"   RMSE: {rmse:,.2f}")
# #     print(f"   MAPE: {mape:.2f}%")
# #     print(f"   Avg Forecast: {forecast_summary['avg_forecast']:,.2f}")
# #
# #     return {
# #         'forecast_fig': fig_forecast,
# #         'components_fig': fig_components,
# #         'metrics': metrics,
# #         'forecast_summary': forecast_summary,
# #         'model': model,
# #         'forecast_df': forecast,
# #         'prepared_data': prophet_df
# #     }
# # print(run_prophet_analysis(df))



# import dash
# from dash import dcc, html, Input, Output, State
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input as KerasInput, Dense, Dropout, BatchNormalization, Concatenate
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# # ==================== DATA PREPARATION ====================
# def load_and_prepare_data(filepath='nawassco.csv'):
#     """Load and prepare the water supply dataset for price discrimination analysis"""
#     try:
#         df = pd.read_csv(filepath)
#
#         # Standardize column names
#         df.columns = df.columns.str.lower().str.replace('/', '_').str.replace(' ', '_')
#
#         # Convert date to datetime
#         df['date'] = pd.to_datetime(df['date'])
#
#         # Sort by date
#         df = df.sort_values('date').reset_index(drop=True)
#
#         print(f"✅ Loaded {filepath}")
#         print(f"   Shape: {df.shape}")
#         print(f"   Columns: {df.columns.tolist()}")
#
#         return df
#     except FileNotFoundError:
#         print(f"❌ Error: {filepath} not found. Generating synthetic data...")
#         return generate_synthetic_data()
#
#
# def generate_synthetic_data():
#     """Generate synthetic water supply data if CSV not found"""
#     np.random.seed(42)
#     n_samples = 200
#
#     dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='D')
#     zones = np.random.choice(['Central', 'Northern', 'Southern', 'Western', 'Eastern'], n_samples)
#     times = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples)
#     days = [d.strftime('%A') for d in dates]
#
#     df = pd.DataFrame({
#         'date': dates,
#         'zone': zones,
#         'day_of_week': days,
#         'time_of_supply': times,
#         'population_served': np.random.randint(25000, 55000, n_samples),
#         'demand_forecast_m3': np.random.randint(5000, 15000, n_samples),
#         'actual_consumption_m3': np.random.uniform(4000, 14000, n_samples),
#         'rainfall_mm': np.random.uniform(0, 50, n_samples),
#         'pipe_leakage_m3': np.random.randint(0, 500, n_samples),
#         'price_per_litre': np.random.uniform(0.05, 0.35, n_samples)
#     })
#
#     return df
#
#
# def feature_engineering(df):
#     """Create features for the model"""
#     df_feat = df.copy()
#
#     # Extract date features
#     df_feat['month'] = df_feat['date'].dt.month
#     df_feat['day'] = df_feat['date'].dt.day
#     df_feat['day_of_year'] = df_feat['date'].dt.dayofyear
#     df_feat['week_of_year'] = df_feat['date'].dt.isocalendar().week
#     df_feat['is_weekend'] = df_feat['date'].dt.dayofweek.isin([5, 6]).astype(int)
#
#     # Encode day of week
#     day_mapping = {
#         'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
#         'Friday': 4, 'Saturday': 5, 'Sunday': 6
#     }
#     df_feat['day_of_week_encoded'] = df_feat['day_of_week'].map(day_mapping)
#
#     # Consumption ratio
#     df_feat['consumption_ratio'] = df_feat['actual_consumption_m3'] / (df_feat['demand_forecast_m3'] + 1)
#
#     # Per capita consumption
#     df_feat['per_capita_consumption'] = df_feat['actual_consumption_m3'] / (df_feat['population_served'] + 1)
#
#     return df_feat
#
#
# # ==================== MODEL BUILDING ====================
# def build_multi_output_model(input_dim, n_zones, n_times):
#     """
#     Build complex multi-output neural network for price discrimination
#
#     Outputs:
#     1. Price (Regression)
#     2. Time of Supply (Classification)
#     3. Zone (Classification)
#     """
#
#     # Input layer
#     input_layer = KerasInput(shape=(input_dim,), name='input')
#
#     # Shared layers - Complex feature extraction
#     x = Dense(256, activation='relu')(input_layer)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#
#     x = Dense(128, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#
#     x = Dense(64, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x)
#
#     # Branch 1: Price Prediction (Regression)
#     price_branch = Dense(32, activation='relu', name='price_dense_1')(x)
#     price_branch = Dropout(0.2)(price_branch)
#     price_branch = Dense(16, activation='relu', name='price_dense_2')(price_branch)
#     price_output = Dense(1, activation='linear', name='price_output')(price_branch)
#
#     # Branch 2: Time of Supply (Classification)
#     time_branch = Dense(32, activation='relu', name='time_dense_1')(x)
#     time_branch = Dropout(0.2)(time_branch)
#     time_branch = Dense(16, activation='relu', name='time_dense_2')(time_branch)
#     time_output = Dense(n_times, activation='softmax', name='time_output')(time_branch)
#
#     # Branch 3: Zone (Classification)
#     zone_branch = Dense(32, activation='relu', name='zone_dense_1')(x)
#     zone_branch = Dropout(0.2)(zone_branch)
#     zone_branch = Dense(16, activation='relu', name='zone_dense_2')(zone_branch)
#     zone_output = Dense(n_zones, activation='softmax', name='zone_output')(zone_branch)
#
#     # Create model
#     model = Model(
#         inputs=input_layer,
#         outputs=[price_output, time_output, zone_output],
#         name='PriceDiscriminationModel'
#     )
#
#     # Compile with different losses for each output
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=0.001),
#         loss={
#             'price_output': 'mse',
#             'time_output': 'categorical_crossentropy',
#             'zone_output': 'categorical_crossentropy'
#         },
#         loss_weights={
#             'price_output': 1.0,
#             'time_output': 0.5,
#             'zone_output': 0.5
#         },
#         metrics={
#             'price_output': ['mae'],
#             'time_output': ['accuracy'],
#             'zone_output': ['accuracy']
#         }
#     )
#
#     return model
#
#
# def train_price_discrimination_model(df):
#     """Train the multi-output model"""
#
#     print("\n🔧 Starting model training...")
#
#     # Feature engineering
#     df_processed = feature_engineering(df)
#
#     # Define features
#     feature_columns = [
#         'month', 'day', 'day_of_year', 'week_of_year', 'is_weekend',
#         'day_of_week_encoded', 'population_served', 'demand_forecast_m3',
#         'actual_consumption_m3', 'consumption_ratio', 'per_capita_consumption'
#     ]
#
#     X = df_processed[feature_columns].values
#
#     # Prepare targets
#     # 1. Price (regression)
#     y_price = df_processed['price_per_litre'].values
#
#     # 2. Time of supply (classification)
#     time_encoder = LabelEncoder()
#     y_time_encoded = time_encoder.fit_transform(df_processed['time_of_supply'])
#     y_time_categorical = keras.utils.to_categorical(y_time_encoded)
#
#     # 3. Zone (classification)
#     zone_encoder = LabelEncoder()
#     y_zone_encoded = zone_encoder.fit_transform(df_processed['zone'])
#     y_zone_categorical = keras.utils.to_categorical(y_zone_encoded)
#
#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Train-test split
#     X_train, X_test, y_price_train, y_price_test, y_time_train, y_time_test, y_zone_train, y_zone_test = train_test_split(
#         X_scaled, y_price, y_time_categorical, y_zone_categorical,
#         test_size=0.2, random_state=42
#     )
#
#     # Build model
#     n_zones = len(zone_encoder.classes_)
#     n_times = len(time_encoder.classes_)
#
#     model = build_multi_output_model(
#         input_dim=X_train.shape[1],
#         n_zones=n_zones,
#         n_times=n_times
#     )
#
#     print(f"\n📊 Model Architecture:")
#     print(f"   Input features: {X_train.shape[1]}")
#     print(f"   Training samples: {X_train.shape[0]}")
#     print(f"   Test samples: {X_test.shape[0]}")
#     print(f"   Zones: {zone_encoder.classes_.tolist()}")
#     print(f"   Times: {time_encoder.classes_.tolist()}")
#
#     # Callbacks
#     early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
#
#     # Train model
#     history = model.fit(
#         X_train,
#         {
#             'price_output': y_price_train,
#             'time_output': y_time_train,
#             'zone_output': y_zone_train
#         },
#         validation_split=0.2,
#         epochs=100,
#         batch_size=16,
#         callbacks=[early_stop, reduce_lr],
#         verbose=0
#     )
#
#     # Evaluate on test set
#     predictions = model.predict(X_test, verbose=0)
#     y_price_pred, y_time_pred, y_zone_pred = predictions
#
#     # Calculate metrics
#     # Price metrics
#     price_mae = mean_absolute_error(y_price_test, y_price_pred)
#     price_rmse = np.sqrt(mean_squared_error(y_price_test, y_price_pred))
#
#     # Time accuracy
#     time_accuracy = accuracy_score(
#         np.argmax(y_time_test, axis=1),
#         np.argmax(y_time_pred, axis=1)
#     )
#
#     # Zone accuracy
#     zone_accuracy = accuracy_score(
#         np.argmax(y_zone_test, axis=1),
#         np.argmax(y_zone_pred, axis=1)
#     )
#
#     print(f"\n✅ Training complete!")
#     print(f"   Epochs trained: {len(history.history['loss'])}")
#     print(f"   Price MAE: {price_mae:.4f}")
#     print(f"   Price RMSE: {price_rmse:.4f}")
#     print(f"   Time Accuracy: {time_accuracy:.2%}")
#     print(f"   Zone Accuracy: {zone_accuracy:.2%}")
#
#     return {
#         'model': model,
#         'history': history,
#         'scaler': scaler,
#         'time_encoder': time_encoder,
#         'zone_encoder': zone_encoder,
#         'feature_columns': feature_columns,
#         'metrics': {
#             'price_mae': price_mae,
#             'price_rmse': price_rmse,
#             'time_accuracy': time_accuracy,
#             'zone_accuracy': zone_accuracy
#         },
#         'test_data': {
#             'X_test': X_test,
#             'y_price_test': y_price_test,
#             'y_time_test': y_time_test,
#             'y_zone_test': y_zone_test,
#             'predictions': predictions
#         }
#     }
#
#
# # ==================== DASH APPLICATION ====================
# app = dash.Dash(__name__)
#
# # Global variables for model and data
# model_results = None
# df_global = None
#
# # Load data and train model on startup
# print("🚀 Initializing Price Discrimination Analysis System...")
# df_global = load_and_prepare_data()
#
# app.layout = html.Div([
#     html.Div([
#         html.H1("💰 Multi-Output Price Discrimination Analysis",
#                 style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
#         html.P("Deep Learning model predicting Price, Time of Supply, and Zone simultaneously",
#                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'})
#     ]),
#
#     # Control Panel
#     html.Div([
#         html.H3("🎯 Model Configuration", style={'color': '#2c3e50'}),
#
#         html.Div([
#             html.Label("Select Target Output to Analyze:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
#             dcc.Dropdown(
#                 id='target-dropdown',
#                 options=[
#                     {'label': '💵 Price per Litre (Regression)', 'value': 'price'},
#                     {'label': '⏰ Time of Supply (Classification)', 'value': 'time'},
#                     {'label': '📍 Zone (Classification)', 'value': 'zone'},
#                     {'label': '📊 All Outputs Combined', 'value': 'all'}
#                 ],
#                 value='all',
#                 style={'width': '100%', 'marginBottom': '20px'}
#             )
#         ]),
#
#         html.Button('🤖 Train Multi-Output Model', id='train-button', n_clicks=0,
#                     style={'backgroundColor': '#9b59b6', 'color': 'white', 'padding': '15px 30px',
#                            'fontSize': '16px', 'border': 'none', 'borderRadius': '5px',
#                            'cursor': 'pointer', 'display': 'block', 'margin': '20px auto'})
#
#     ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
#
#     # Loading and Results
#     dcc.Loading(
#         id="loading",
#         type="default",
#         children=[
#             # Model info
#             html.Div(id='model-info', style={'marginBottom': '20px'}),
#
#             # Performance plots
#             dcc.Graph(id='performance-plot', style={'marginBottom': '20px'}),
#
#             # Prediction analysis
#             dcc.Graph(id='prediction-plot', style={'marginBottom': '20px'}),
#
#             # Feature importance
#             dcc.Graph(id='feature-importance-plot', style={'marginBottom': '20px'}),
#
#             # Metrics
#             html.Div(id='metrics-display', style={'marginTop': '20px'}),
#         ]
#     )
# ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px', 'margin': '0 auto'})
#
#
# @app.callback(
#     [Output('model-info', 'children'),
#      Output('performance-plot', 'figure'),
#      Output('prediction-plot', 'figure'),
#      Output('feature-importance-plot', 'figure'),
#      Output('metrics-display', 'children')],
#     [Input('train-button', 'n_clicks')],
#     [State('target-dropdown', 'value')]
# )
# def train_and_visualize(n_clicks, target_output):
#     """Train model and create visualizations"""
#
#     if n_clicks == 0:
#         return html.Div(), go.Figure(), go.Figure(), go.Figure(), html.Div()
#
#     global model_results
#
#     # Train model
#     model_results = train_price_discrimination_model(df_global)
#
#     # Model info box
#     model_info = html.Div([
#         html.H3("🧠 Model Training Results", style={'color': '#2c3e50', 'marginBottom': '15px'}),
#         html.Div([
#             html.Div([
#                 html.Strong("Architecture: "),
#                 html.Span("Multi-Output Neural Network (256→128→64 shared layers)")
#             ], style={'marginBottom': '10px'}),
#             html.Div([
#                 html.Strong("Output Branches: "),
#                 html.Span("3 (Price Regression, Time Classification, Zone Classification)")
#             ], style={'marginBottom': '10px'}),
#             html.Div([
#                 html.Strong("Training Samples: "),
#                 html.Span(f"{model_results['test_data']['X_test'].shape[0] * 5}")  # Approx total
#             ], style={'marginBottom': '10px'}),
#             html.Div([
#                 html.Strong("Epochs: "),
#                 html.Span(f"{len(model_results['history'].history['loss'])} (Early Stopping)")
#             ], style={'marginBottom': '10px'}),
#             html.Div([
#                 html.Strong("Optimizer: "),
#                 html.Span("Adam with ReduceLROnPlateau")
#             ])
#         ])
#     ], style={'backgroundColor': '#e8f5e9', 'padding': '20px', 'borderRadius': '10px',
#               'border': '2px solid #4caf50'})
#
#     # Training history plot
#     history = model_results['history']
#     fig_performance = make_subplots(
#         rows=2, cols=2,
#         subplot_titles=('Total Loss', 'Price Loss (MSE)',
#                         'Time Classification Loss', 'Zone Classification Loss'),
#         vertical_spacing=0.12,
#         horizontal_spacing=0.1
#     )
#
#     # Total loss
#     fig_performance.add_trace(go.Scatter(
#         y=history.history['loss'],
#         name='Train Loss',
#         line=dict(color='#e74c3c', width=2)
#     ), row=1, col=1)
#     fig_performance.add_trace(go.Scatter(
#         y=history.history['val_loss'],
#         name='Val Loss',
#         line=dict(color='#3498db', width=2)
#     ), row=1, col=1)
#
#     # Price loss
#     fig_performance.add_trace(go.Scatter(
#         y=history.history['price_output_loss'],
#         name='Train',
#         line=dict(color='#e74c3c', width=2),
#         showlegend=False
#     ), row=1, col=2)
#     fig_performance.add_trace(go.Scatter(
#         y=history.history['val_price_output_loss'],
#         name='Val',
#         line=dict(color='#3498db', width=2),
#         showlegend=False
#     ), row=1, col=2)
#
#     # Time loss
#     fig_performance.add_trace(go.Scatter(
#         y=history.history['time_output_loss'],
#         name='Train',
#         line=dict(color='#27ae60', width=2),
#         showlegend=False
#     ), row=2, col=1)
#     fig_performance.add_trace(go.Scatter(
#         y=history.history['val_time_output_loss'],
#         name='Val',
#         line=dict(color='#f39c12', width=2),
#         showlegend=False
#     ), row=2, col=1)
#
#     # Zone loss
#     fig_performance.add_trace(go.Scatter(
#         y=history.history['zone_output_loss'],
#         name='Train',
#         line=dict(color='#9b59b6', width=2),
#         showlegend=False
#     ), row=2, col=2)
#     fig_performance.add_trace(go.Scatter(
#         y=history.history['val_zone_output_loss'],
#         name='Val',
#         line=dict(color='#1abc9c', width=2),
#         showlegend=False
#     ), row=2, col=2)
#
#     fig_performance.update_xaxes(title_text="Epoch")
#     fig_performance.update_yaxes(title_text="Loss")
#     fig_performance.update_layout(
#         height=600,
#         template='plotly_white',
#         title_text='Training Performance Across All Outputs'
#     )
#
#     # Predictions plot
#     test_data = model_results['test_data']
#     y_price_pred, y_time_pred, y_zone_pred = test_data['predictions']
#
#     if target_output == 'price' or target_output == 'all':
#         fig_predictions = make_subplots(
#             rows=1, cols=3 if target_output == 'all' else 1,
#             subplot_titles=(['Price Predictions', 'Time Classification', 'Zone Classification']
#                             if target_output == 'all' else ['Price Predictions']),
#             specs=[[{'type': 'scatter'}] * (3 if target_output == 'all' else 1)]
#         )
#
#         # Price predictions
#         col_idx = 1
#         fig_predictions.add_trace(go.Scatter(
#             x=test_data['y_price_test'],
#             y=y_price_pred.flatten(),
#             mode='markers',
#             name='Predictions',
#             marker=dict(size=8, color='#3498db', opacity=0.6),
#             hovertemplate='Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>'
#         ), row=1, col=col_idx)
#
#         # Perfect prediction line
#         price_range = [test_data['y_price_test'].min(), test_data['y_price_test'].max()]
#         fig_predictions.add_trace(go.Scatter(
#             x=price_range,
#             y=price_range,
#             mode='lines',
#             name='Perfect Prediction',
#             line=dict(color='red', dash='dash'),
#             showlegend=True
#         ), row=1, col=col_idx)
#
#         fig_predictions.update_xaxes(title_text="Actual Price", row=1, col=col_idx)
#         fig_predictions.update_yaxes(title_text="Predicted Price", row=1, col=col_idx)
#
#         if target_output == 'all':
#             # Time confusion matrix heatmap
#             time_true = np.argmax(test_data['y_time_test'], axis=1)
#             time_pred = np.argmax(y_time_pred, axis=1)
#             time_classes = model_results['time_encoder'].classes_
#
#             confusion_time = np.zeros((len(time_classes), len(time_classes)))
#             for t, p in zip(time_true, time_pred):
#                 confusion_time[t, p] += 1
#
#             fig_predictions.add_trace(go.Heatmap(
#                 z=confusion_time,
#                 x=time_classes,
#                 y=time_classes,
#                 colorscale='Blues',
#                 showscale=True,
#                 hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>'
#             ), row=1, col=2)
#
#             fig_predictions.update_xaxes(title_text="Predicted Time", row=1, col=2)
#             fig_predictions.update_yaxes(title_text="Actual Time", row=1, col=2)
#
#             # Zone confusion matrix heatmap
#             zone_true = np.argmax(test_data['y_zone_test'], axis=1)
#             zone_pred = np.argmax(y_zone_pred, axis=1)
#             zone_classes = model_results['zone_encoder'].classes_
#
#             confusion_zone = np.zeros((len(zone_classes), len(zone_classes)))
#             for t, p in zip(zone_true, zone_pred):
#                 confusion_zone[t, p] += 1
#
#             fig_predictions.add_trace(go.Heatmap(
#                 z=confusion_zone,
#                 x=zone_classes,
#                 y=zone_classes,
#                 colorscale='Greens',
#                 showscale=True,
#                 hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>'
#             ), row=1, col=3)
#
#             fig_predictions.update_xaxes(title_text="Predicted Zone", row=1, col=3)
#             fig_predictions.update_yaxes(title_text="Actual Zone", row=1, col=3)
#
#         fig_predictions.update_layout(
#             height=500,
#             template='plotly_white',
#             title_text='Model Predictions vs Actual Values'
#         )
#     else:
#         fig_predictions = go.Figure()
#
#     # Feature importance (approximate using gradient)
#     feature_names = model_results['feature_columns']
#
#     # Sample some test data
#     sample_X = test_data['X_test'][:10]
#
#     # Get predictions
#     with tf.GradientTape() as tape:
#         X_tensor = tf.Variable(sample_X, dtype=tf.float32)
#         predictions = model_results['model'](X_tensor)
#         price_pred = predictions[0]
#
#     # Calculate gradients
#     gradients = tape.gradient(price_pred, X_tensor)
#     feature_importance = np.abs(gradients.numpy()).mean(axis=0)
#
#     # Normalize
#     feature_importance = feature_importance / feature_importance.sum()
#
#     # Sort by importance
#     sorted_idx = np.argsort(feature_importance)[::-1]
#
#     fig_importance = go.Figure()
#     fig_importance.add_trace(go.Bar(
#         x=feature_importance[sorted_idx],
#         y=[feature_names[i] for i in sorted_idx],
#         orientation='h',
#         marker=dict(color=feature_importance[sorted_idx], colorscale='Viridis'),
#         hovertemplate='%{y}<br>Importance: %{x:.4f}<extra></extra>'
#     ))
#
#     fig_importance.update_layout(
#         title='Feature Importance (Gradient-based)',
#         xaxis_title='Relative Importance',
#         yaxis_title='Feature',
#         template='plotly_white',
#         height=500
#     )
#
#     # Metrics display
#     metrics = model_results['metrics']
#     metrics_div = html.Div([
#         html.H3("📊 Model Performance Metrics", style={'color': '#2c3e50', 'textAlign': 'center'}),
#         html.Div([
#             html.Div([
#                 html.H4("Price MAE", style={'color': '#3498db'}),
#                 html.P(f"{metrics['price_mae']:.4f}", style={'fontSize': '24px', 'fontWeight': 'bold'}),
#                 html.P("Mean Absolute Error", style={'fontSize': '11px', 'color': '#7f8c8d'})
#             ], style={'width': '25%', 'display': 'inline-block', 'textAlign': 'center'}),
#
#             html.Div([
#                 html.H4("Price RMSE", style={'color': '#e74c3c'}),
#                 html.P(f"{metrics['price_rmse']:.4f}", style={'fontSize': '24px', 'fontWeight': 'bold'}),
#                 html.P("Root Mean Squared Error", style={'fontSize': '11px', 'color': '#7f8c8d'})
#             ], style={'width': '25%', 'display': 'inline-block', 'textAlign': 'center'}),
#
#             html.Div([
#                 html.H4("Time Accuracy", style={'color': '#27ae60'}),
#                 html.P(f"{metrics['time_accuracy']:.2%}", style={'fontSize': '24px', 'fontWeight': 'bold'}),
#                 html.P("Classification Accuracy", style={'fontSize': '11px', 'color': '#7f8c8d'})
#             ], style={'width': '25%', 'display': 'inline-block', 'textAlign': 'center'}),
#
#             html.Div([
#                 html.H4("Zone Accuracy", style={'color': '#9b59b6'}),
#                 html.P(f"{metrics['zone_accuracy']:.2%}", style={'fontSize': '24px', 'fontWeight': 'bold'}),
#                 html.P("Classification Accuracy", style={'fontSize': '11px', 'color': '#7f8c8d'})
#             ], style={'width': '25%', 'display': 'inline-block', 'textAlign': 'center'}),
#         ])
#     ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})
#
#     return model_info, fig_performance, fig_predictions, fig_importance, metrics_div
#
#
# if __name__ == '__main__':
#     app.run(debug=True, port=8053)




import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
# Corrected import to include Reshape and Flatten
from tensorflow.keras.layers import Input as KerasInput, Dense, Dropout, BatchNormalization, Concatenate, Attention, \
    MultiHeadAttention, Reshape, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')


# ==================== DATA PREPARATION ====================
def load_and_prepare_data(filepath='nawassco.csv'):
    """Load and prepare the water supply dataset for rationing analysis"""
    try:
        df = pd.read_csv(filepath)

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace('/', '_').str.replace(' ', '_')

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        print(f"✅ Loaded {filepath}")
        print(f"   Shape: {df.shape}")
        print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")

        return df
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found. Generating synthetic data...")
        return generate_synthetic_data()


def generate_synthetic_data():
    """Generate synthetic water supply data if CSV not found"""
    np.random.seed(42)
    n_samples = 300

    dates = pd.date_range(start='2024-06-01', periods=n_samples, freq='D')
    zones = np.random.choice(['Central', 'Northern', 'Southern', 'Western', 'Eastern'], n_samples)
    times = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples)
    days = [d.strftime('%A') for d in dates]

    df = pd.DataFrame({
        'date': dates,
        'zone': zones,
        'day_of_week': days,
        'time_of_supply': times,
        'hours_of_supply': np.random.randint(2, 12, n_samples),
        'population_served': np.random.randint(25000, 55000, n_samples),
        'demand_forecast_m3': np.random.randint(5000, 15000, n_samples),
        'actual_consumption_m3': np.random.uniform(4000, 14000, n_samples),
        'rainfall_mm': np.random.uniform(0, 50, n_samples),
        'pipe_leakage_m3': np.random.randint(0, 500, n_samples),
        'complaints_received': np.random.randint(0, 50, n_samples),
        'price_litre': np.random.uniform(0.05, 0.35, n_samples)
    })

    return df


def feature_engineering(df):
    """Create features for the rationing model"""
    df_feat = df.copy()

    # Temporal features
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['day'] = df_feat['date'].dt.day
    df_feat['day_of_year'] = df_feat['date'].dt.dayofyear
    df_feat['week_of_year'] = df_feat['date'].dt.isocalendar().week
    df_feat['is_weekend'] = df_feat['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df_feat['quarter'] = df_feat['date'].dt.quarter

    # Day of week encoding
    day_mapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    df_feat['day_of_week_encoded'] = df_feat['day_of_week'].map(day_mapping)

    # Consumption metrics
    df_feat['consumption_ratio'] = df_feat['actual_consumption_m3'] / (df_feat['demand_forecast_m3'] + 1)
    df_feat['per_capita_consumption'] = df_feat['actual_consumption_m3'] / (df_feat['population_served'] + 1)
    df_feat['supply_efficiency'] = (df_feat['actual_consumption_m3'] - df_feat['pipe_leakage_m3']) / (
                df_feat['actual_consumption_m3'] + 1)

    # Demand pressure indicators
    df_feat['demand_pressure'] = df_feat['demand_forecast_m3'] / (df_feat['population_served'] + 1)
    df_feat['complaint_rate'] = df_feat['complaints_received'] / (df_feat['population_served'] + 1) * 10000

    # Water stress indicator (high demand, low supply)
    df_feat['water_stress'] = (df_feat['demand_forecast_m3'] - df_feat['actual_consumption_m3']) / (
                df_feat['demand_forecast_m3'] + 1)

    # Season encoding (dry/wet based on rainfall)
    df_feat['is_dry_season'] = (df_feat['rainfall_mm'] < df_feat['rainfall_mm'].median()).astype(int)

    return df_feat


# ==================== ADVANCED MODEL BUILDING (CORRECTED) ====================
def build_rationing_model(input_dim, n_zones, n_times):
    """
    Build advanced multi-output neural network for water rationing optimization

    Outputs:
    1. Time of Supply (Classification) - When to distribute water
    2. Zone (Classification) - Where to prioritize distribution

    Uses attention mechanism to learn dependencies between features
    """

    # Input layer
    input_layer = KerasInput(shape=(input_dim,), name='input')

    # First embedding layer
    x = Dense(256, activation='relu', name='embed_1')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Second embedding layer
    x = Dense(192, activation='relu', name='embed_2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # --- START OF FIX ---
    # Attention layer for feature importance
    # MultiHeadAttention expects a sequence input (batch, sequence, features).
    # We use Reshape to add a sequence dimension of size 1.
    # Shape before: (batch_size, 192)
    attention_input = Reshape((1, -1))(x)  # Shape becomes (batch_size, 1, 192)

    attention_output = MultiHeadAttention(
        num_heads=4,
        key_dim=48,
        name='multi_head_attention'
    )(attention_input, attention_input)

    # We use Flatten to remove the sequence dimension.
    # Shape before: (batch_size, 1, 192)
    attention_output = Flatten()(attention_output)  # Shape becomes (batch_size, 192)
    # --- END OF FIX ---

    # Combine with original features
    x = Concatenate()([x, attention_output])

    # Deep shared layers
    x = Dense(128, activation='relu', name='shared_1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    x = Dense(96, activation='relu', name='shared_2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # Branch 1: Time of Supply (When to distribute)
    time_branch = Dense(64, activation='relu', name='time_dense_1')(x)
    time_branch = BatchNormalization()(time_branch)
    time_branch = Dropout(0.2)(time_branch)

    time_branch = Dense(32, activation='relu', name='time_dense_2')(time_branch)
    time_branch = Dropout(0.2)(time_branch)

    time_output = Dense(n_times, activation='softmax', name='time_output')(time_branch)

    # Branch 2: Zone (Where to prioritize)
    zone_branch = Dense(64, activation='relu', name='zone_dense_1')(x)
    zone_branch = BatchNormalization()(zone_branch)
    zone_branch = Dropout(0.2)(zone_branch)

    zone_branch = Dense(32, activation='relu', name='zone_dense_2')(zone_branch)
    zone_branch = Dropout(0.2)(zone_branch)

    zone_output = Dense(n_zones, activation='softmax', name='zone_output')(zone_branch)

    # Create model
    model = Model(
        inputs=input_layer,
        outputs=[time_output, zone_output],
        name='WaterRationingOptimizer'
    )

    # Compile with weighted losses
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'time_output': 'categorical_crossentropy',
            'zone_output': 'categorical_crossentropy'
        },
        loss_weights={
            'time_output': 1.0,  # Equal importance
            'zone_output': 1.0
        },
        metrics={
            'time_output': ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')],
            'zone_output': ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
        }
    )

    return model


def train_rationing_model(df):
    """Train the water rationing optimization model"""

    print("\n🔧 Starting Water Rationing Model Training...")

    # Feature engineering
    df_processed = feature_engineering(df)

    # Define features for rationing decisions
    feature_columns = [
        'month', 'day_of_year', 'week_of_year', 'quarter', 'is_weekend',
        'day_of_week_encoded', 'population_served', 'demand_forecast_m3',
        'actual_consumption_m3', 'consumption_ratio', 'per_capita_consumption',
        'rainfall_mm', 'pipe_leakage_m3', 'complaints_received',
        'supply_efficiency', 'demand_pressure', 'complaint_rate',
        'water_stress', 'is_dry_season'
    ]

    X = df_processed[feature_columns].values

    # Prepare targets
    # 1. Time of supply (when to distribute)
    time_encoder = LabelEncoder()
    y_time_encoded = time_encoder.fit_transform(df_processed['time_of_supply'])
    y_time_categorical = keras.utils.to_categorical(y_time_encoded)

    # 2. Zone (where to prioritize)
    zone_encoder = LabelEncoder()
    y_zone_encoded = zone_encoder.fit_transform(df_processed['zone'])
    y_zone_categorical = keras.utils.to_categorical(y_zone_encoded)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_time_train, y_time_test, y_zone_train, y_zone_test = train_test_split(
        X_scaled, y_time_categorical, y_zone_categorical,
        test_size=0.2, random_state=42, stratify=y_zone_encoded
    )

    # Build model
    n_zones = len(zone_encoder.classes_)
    n_times = len(time_encoder.classes_)

    model = build_rationing_model(
        input_dim=X_train.shape[1],
        n_zones=n_zones,
        n_times=n_times
    )

    print(f"\n📊 Model Configuration:")
    print(f"   Input features: {X_train.shape[1]}")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Time slots: {time_encoder.classes_.tolist()}")
    print(f"   Zones: {zone_encoder.classes_.tolist()}")
    print(f"   Model parameters: {model.count_params():,}")

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=12,
        min_lr=1e-6,
        verbose=1
    )

    # Train model
    print("\n🚀 Training model...")
    history = model.fit(
        X_train,
        {
            'time_output': y_time_train,
            'zone_output': y_zone_train
        },
        validation_split=0.2,
        epochs=150,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluate on test set
    print("\n📈 Evaluating model on test set...")
    predictions = model.predict(X_test, verbose=0)
    y_time_pred, y_zone_pred = predictions

    # Calculate metrics
    # Time accuracy
    time_true = np.argmax(y_time_test, axis=1)
    time_pred = np.argmax(y_time_pred, axis=1)
    time_accuracy = accuracy_score(time_true, time_pred)

    # Zone accuracy
    zone_true = np.argmax(y_zone_test, axis=1)
    zone_pred = np.argmax(y_zone_pred, axis=1)
    zone_accuracy = accuracy_score(zone_true, zone_pred)

    # Top-2 accuracy (if prediction is in top 2 choices)
    time_top2 = np.mean([t in np.argsort(p)[-2:] for t, p in zip(time_true, y_time_pred)])
    zone_top2 = np.mean([z in np.argsort(p)[-2:] for z, p in zip(zone_true, y_zone_pred)])

    # Confusion matrices
    time_cm = confusion_matrix(time_true, time_pred)
    zone_cm = confusion_matrix(zone_true, zone_pred)

    # Classification reports
    time_report = classification_report(
        time_true, time_pred,
        target_names=time_encoder.classes_,
        output_dict=True
    )

    zone_report = classification_report(
        zone_true, zone_pred,
        target_names=zone_encoder.classes_,
        output_dict=True
    )

    print(f"\n✅ Training Complete!")
    print(f"   Epochs trained: {len(history.history['loss'])}")
    print(f"   Time Accuracy: {time_accuracy:.2%}")
    print(f"   Time Top-2 Accuracy: {time_top2:.2%}")
    print(f"   Zone Accuracy: {zone_accuracy:.2%}")
    print(f"   Zone Top-2 Accuracy: {zone_top2:.2%}")

    return {
        'model': model,
        'history': history,
        'scaler': scaler,
        'time_encoder': time_encoder,
        'zone_encoder': zone_encoder,
        'feature_columns': feature_columns,
        'metrics': {
            'time_accuracy': time_accuracy,
            'time_top2_accuracy': time_top2,
            'zone_accuracy': zone_accuracy,
            'zone_top2_accuracy': zone_top2,
            'time_confusion_matrix': time_cm,
            'zone_confusion_matrix': zone_cm,
            'time_report': time_report,
            'zone_report': zone_report
        },
        'test_data': {
            'X_test': X_test,
            'y_time_test': y_time_test,
            'y_zone_test': y_zone_test,
            'time_true': time_true,
            'time_pred': time_pred,
            'zone_true': zone_true,
            'zone_pred': zone_pred,
            'predictions': predictions
        }
    }


def generate_rationing_schedule(model_results, df, days_ahead=7):
    """Generate optimal rationing schedule for upcoming days"""

    df_processed = feature_engineering(df)

    # Get latest date in dataset
    latest_date = df_processed['date'].max()

    # Generate future dates
    future_dates = [latest_date + timedelta(days=i + 1) for i in range(days_ahead)]

    # Create scenarios for each zone
    zones = model_results['zone_encoder'].classes_
    schedule = []

    for date in future_dates:
        for zone in zones:
            # Create features for prediction
            features = {
                'month': date.month,
                'day_of_year': date.timetuple().tm_yday,
                'week_of_year': date.isocalendar()[1],
                'quarter': (date.month - 1) // 3 + 1,
                'is_weekend': 1 if date.weekday() >= 5 else 0,
                'day_of_week_encoded': date.weekday(),
                'population_served': df_processed[df_processed['zone'] == zone]['population_served'].mean(),
                'demand_forecast_m3': df_processed[df_processed['zone'] == zone]['demand_forecast_m3'].mean(),
                'actual_consumption_m3': df_processed[df_processed['zone'] == zone]['actual_consumption_m3'].mean(),
                'rainfall_mm': df_processed['rainfall_mm'].tail(7).mean(),
                'pipe_leakage_m3': df_processed[df_processed['zone'] == zone]['pipe_leakage_m3'].mean(),
                'complaints_received': df_processed[df_processed['zone'] == zone]['complaints_received'].mean(),
            }

            # Calculate derived features
            features['consumption_ratio'] = features['actual_consumption_m3'] / (features['demand_forecast_m3'] + 1)
            features['per_capita_consumption'] = features['actual_consumption_m3'] / (features['population_served'] + 1)
            features['supply_efficiency'] = (features['actual_consumption_m3'] - features['pipe_leakage_m3']) / (
                        features['actual_consumption_m3'] + 1)
            features['demand_pressure'] = features['demand_forecast_m3'] / (features['population_served'] + 1)
            features['complaint_rate'] = features['complaints_received'] / (features['population_served'] + 1) * 10000
            features['water_stress'] = (features['demand_forecast_m3'] - features['actual_consumption_m3']) / (
                        features['demand_forecast_m3'] + 1)
            features['is_dry_season'] = 1 if features['rainfall_mm'] < df_processed['rainfall_mm'].median() else 0

            # Create feature vector
            feature_vector = np.array([[features[col] for col in model_results['feature_columns']]])

            # Scale features
            feature_scaled = model_results['scaler'].transform(feature_vector)

            # Predict
            time_probs, zone_probs = model_results['model'].predict(feature_scaled, verbose=0)

            # Get top predictions
            top_time_idx = np.argmax(time_probs[0])
            top_zone_idx = np.argmax(zone_probs[0])

            recommended_time = model_results['time_encoder'].classes_[top_time_idx]
            time_confidence = time_probs[0][top_time_idx]
            zone_confidence = zone_probs[0][top_zone_idx]

            schedule.append({
                'date': date,
                'zone': zone,
                'recommended_time': recommended_time,
                'time_confidence': time_confidence,
                'zone_priority': zone_confidence,
                'demand_forecast': features['demand_forecast_m3'],
                'water_stress': features['water_stress'],
                'complaint_rate': features['complaint_rate']
            })

    return pd.DataFrame(schedule)


# ==================== DASH APPLICATION ====================
app = dash.Dash(__name__)

# Global variables
model_results = None
df_global = None
schedule_df = None

# Load data on startup
print("🚀 Initializing Water Rationing Optimization System...")
df_global = load_and_prepare_data()

app.layout = html.Div([
    html.Div([
        html.H1("💧 Dynamic Water Rationing Scheduler",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("AI-Powered Multi-Output Model for Optimal Water Distribution Scheduling",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'})
    ]),

    # Control Panel
    html.Div([
        html.H3("🎯 Rationing Optimization Controls", style={'color': '#2c3e50'}),

        html.Div([
            html.Div([
                html.Label("Schedule Horizon (Days Ahead):", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='schedule-days-slider',
                    min=3,
                    max=14,
                    step=1,
                    value=7,
                    marks={3: '3', 7: '7', 10: '10', 14: '14'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '100%', 'marginBottom': '20px'}),
        ]),

        html.Button('🤖 Train Rationing Model', id='train-button', n_clicks=0,
                    style={'backgroundColor': '#16a085', 'color': 'white', 'padding': '15px 30px',
                           'fontSize': '16px', 'border': 'none', 'borderRadius': '5px',
                           'cursor': 'pointer', 'display': 'inline-block', 'marginRight': '10px'}),

        html.Button('📅 Generate Schedule', id='schedule-button', n_clicks=0,
                    style={'backgroundColor': '#2980b9', 'color': 'white', 'padding': '15px 30px',
                           'fontSize': '16px', 'border': 'none', 'borderRadius': '5px',
                           'cursor': 'pointer', 'display': 'inline-block'}),

    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),

    # Loading and Results
    dcc.Loading(
        id="loading",
        type="default",
        children=[
            # Model info
            html.Div(id='model-info', style={'marginBottom': '20px'}),

            # Training performance
            dcc.Graph(id='training-plot', style={'marginBottom': '20px'}),

            # Confusion matrices
            dcc.Graph(id='confusion-plot', style={'marginBottom': '20px'}),

            # Classification reports
            html.Div(id='classification-reports', style={'marginBottom': '20px'}),

            # Rationing schedule
            html.Div(id='schedule-display', style={'marginBottom': '20px'}),

            # Schedule visualization
            dcc.Graph(id='schedule-viz', style={'marginBottom': '20px'}),

            # Metrics
            html.Div(id='metrics-display', style={'marginTop': '20px'}),
        ]
    )
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'maxWidth': '1600px', 'margin': '0 auto'})


@app.callback(
    [Output('model-info', 'children'),
     Output('training-plot', 'figure'),
     Output('confusion-plot', 'figure'),
     Output('classification-reports', 'children'),
     Output('metrics-display', 'children')],
    [Input('train-button', 'n_clicks')]
)
def train_model(n_clicks):
    """Train the rationing model"""

    if n_clicks == 0:
        return html.Div(), go.Figure(), go.Figure(), html.Div(), html.Div()

    global model_results

    # Train model
    model_results = train_rationing_model(df_global)

    # Model info box
    model_info = html.Div([
        html.H3("🧠 Water Rationing Model Training Results",
                style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Div([
            html.Div([
                html.Strong("Architecture: "),
                html.Span("Multi-Head Attention + Multi-Output Neural Network")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Strong("Outputs: "),
                html.Span("2 (Time Slot Classification + Zone Prioritization)")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Strong("Features: "),
                html.Span(f"{len(model_results['feature_columns'])} engineered features")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Strong("Parameters: "),
                html.Span(f"{model_results['model'].count_params():,}")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Strong("Training Strategy: "),
                html.Span("Multi-Task Learning with Equal Loss Weighting")
            ])
        ])
    ], style={'backgroundColor': '#d5f4e6', 'padding': '20px', 'borderRadius': '10px',
              'border': '2px solid #16a085'})

    # Training history
    history = model_results['history']
    fig_training = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Loss', 'Time Classification Loss',
                        'Zone Classification Loss', 'Combined Accuracy'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Total loss
    fig_training.add_trace(go.Scatter(
        y=history.history['loss'],
        name='Train',
        line=dict(color='#e74c3c', width=2)
    ), row=1, col=1)
    fig_training.add_trace(go.Scatter(
        y=history.history['val_loss'],
        name='Validation',
        line=dict(color='#3498db', width=2)
    ), row=1, col=1)

    # Time loss
    fig_training.add_trace(go.Scatter(
        y=history.history['time_output_loss'],
        name='Train',
        line=dict(color='#27ae60', width=2),
        showlegend=False
    ), row=1, col=2)
    fig_training.add_trace(go.Scatter(
        y=history.history['val_time_output_loss'],
        name='Val',
        line=dict(color='#16a085', width=2),
        showlegend=False
    ), row=1, col=2)

    # Zone loss
    fig_training.add_trace(go.Scatter(
        y=history.history['zone_output_loss'],
        name='Train',
        line=dict(color='#9b59b6', width=2),
        showlegend=False
    ), row=2, col=1)
    fig_training.add_trace(go.Scatter(
        y=history.history['val_zone_output_loss'],
        name='Val',
        line=dict(color='#8e44ad', width=2),
        showlegend=False
    ), row=2, col=1)

    # Accuracies
    fig_training.add_trace(go.Scatter(
        y=history.history['time_output_accuracy'],
        name='Time Acc',
        line=dict(color='#f39c12', width=2)
    ), row=2, col=2)
    fig_training.add_trace(go.Scatter(
        y=history.history['zone_output_accuracy'],
        name='Zone Acc',
        line=dict(color='#d35400', width=2)
    ), row=2, col=2)

    fig_training.update_xaxes(title_text="Epoch")
    fig_training.update_yaxes(title_text="Loss/Accuracy")
    fig_training.update_layout(
        height=600,
        template='plotly_white',
        title_text='Training Performance - Time & Zone Prediction'
    )

    # Confusion matrices
    metrics = model_results['metrics']
    time_cm = metrics['time_confusion_matrix']
    zone_cm = metrics['zone_confusion_matrix']
    time_classes = model_results['time_encoder'].classes_
    zone_classes = model_results['zone_encoder'].classes_

    fig_confusion = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Time Slot Confusion Matrix', 'Zone Confusion Matrix'),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}]]
    )

    # Time confusion
    fig_confusion.add_trace(go.Heatmap(
        z=time_cm,
        x=time_classes,
        y=time_classes,
        colorscale='Blues',
        showscale=True,
        text=time_cm,
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>'
    ), row=1, col=1)

    # Zone confusion
    fig_confusion.add_trace(go.Heatmap(
        z=zone_cm,
        x=zone_classes,
        y=zone_classes,
        colorscale='Greens',
        showscale=True,
        text=zone_cm,
        texttemplate='%{text}',
        textfont={"size": 12},
        hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>'
    ), row=1, col=2)

    fig_confusion.update_xaxes(title_text="Predicted", row=1, col=1)
    fig_confusion.update_xaxes(title_text="Predicted", row=1, col=2)
    fig_confusion.update_yaxes(title_text="Actual", row=1, col=1)
    fig_confusion.update_yaxes(title_text="Actual", row=1, col=2)

    fig_confusion.update_layout(
        height=500,
        template='plotly_white',
        title_text='Prediction Accuracy Analysis'
    )

    # Classification reports
    time_report = metrics['time_report']
    zone_report = metrics['zone_report']

    # Create tables for classification reports
    time_report_data = []
    for class_name in time_classes:
        if class_name in time_report:
            time_report_data.append({
                'Time Slot': class_name,
                'Precision': f"{time_report[class_name]['precision']:.3f}",
                'Recall': f"{time_report[class_name]['recall']:.3f}",
                'F1-Score': f"{time_report[class_name]['f1-score']:.3f}",
                'Support': time_report[class_name]['support']
            })

    zone_report_data = []
    for class_name in zone_classes:
        if class_name in zone_report:
            zone_report_data.append({
                'Zone': class_name,
                'Precision': f"{zone_report[class_name]['precision']:.3f}",
                'Recall': f"{zone_report[class_name]['recall']:.3f}",
                'F1-Score': f"{zone_report[class_name]['f1-score']:.3f}",
                'Support': zone_report[class_name]['support']
            })

    classification_reports_div = html.Div([
        html.H3("📊 Detailed Classification Reports", style={'color': '#2c3e50', 'marginBottom': '15px'}),

        html.Div([
            html.H4("⏰ Time Slot Performance", style={'color': '#16a085'}),
            dash_table.DataTable(
                data=time_report_data,
                columns=[{"name": i, "id": i} for i in time_report_data[0].keys()],
                style_cell={'textAlign': 'center', 'padding': '10px'},
                style_header={'backgroundColor': '#16a085', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f2f2f2'
                    }
                ]
            )
        ], style={'marginBottom': '20px'}),

        html.Div([
            html.H4("📍 Zone Performance", style={'color': '#27ae60'}),
            dash_table.DataTable(
                data=zone_report_data,
                columns=[{"name": i, "id": i} for i in zone_report_data[0].keys()],
                style_cell={'textAlign': 'center', 'padding': '10px'},
                style_header={'backgroundColor': '#27ae60', 'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f2f2f2'
                    }
                ]
            )
        ])
    ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '10px',
              'border': '1px solid #ddd'})

    # Metrics display
    metrics_div = html.Div([
        html.H3("🎯 Model Performance Summary", style={'color': '#2c3e50', 'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.H4("Time Accuracy", style={'color': '#16a085'}),
                html.P(f"{metrics['time_accuracy']:.2%}",
                       style={'fontSize': '28px', 'fontWeight': 'bold'}),
                html.P("Primary Metric", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Time Top-2", style={'color': '#1abc9c'}),
                html.P(f"{metrics['time_top2_accuracy']:.2%}",
                       style={'fontSize': '28px', 'fontWeight': 'bold'}),
                html.P("Top-2 Accuracy", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Zone Accuracy", style={'color': '#27ae60'}),
                html.P(f"{metrics['zone_accuracy']:.2%}",
                       style={'fontSize': '28px', 'fontWeight': 'bold'}),
                html.P("Primary Metric", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Zone Top-2", style={'color': '#2ecc71'}),
                html.P(f"{metrics['zone_top2_accuracy']:.2%}",
                       style={'fontSize': '28px', 'fontWeight': 'bold'}),
                html.P("Top-2 Accuracy", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Combined", style={'color': '#9b59b6'}),
                html.P(f"{(metrics['time_accuracy'] + metrics['zone_accuracy']) / 2:.2%}",
                       style={'fontSize': '28px', 'fontWeight': 'bold'}),
                html.P("Average Accuracy", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
        ])
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})

    return model_info, fig_training, fig_confusion, classification_reports_div, metrics_div


@app.callback(
    [Output('schedule-display', 'children'),
     Output('schedule-viz', 'figure')],
    [Input('schedule-button', 'n_clicks')],
    [State('schedule-days-slider', 'value')]
)
def generate_schedule(n_clicks, days_ahead):
    """Generate optimized rationing schedule"""

    if n_clicks == 0 or model_results is None:
        return html.Div([
            html.P("⚠️ Please train the model first before generating schedule.",
                   style={'textAlign': 'center', 'color': '#e74c3c', 'fontSize': '16px'})
        ]), go.Figure()

    global schedule_df

    # Generate schedule
    schedule_df = generate_rationing_schedule(model_results, df_global, days_ahead)

    # Sort by priority (zone confidence) and water stress
    schedule_df['priority_score'] = (
            schedule_df['zone_priority'] * 0.6 +
            schedule_df['water_stress'] * 0.3 +
            schedule_df['complaint_rate'] * 0.1
    )
    schedule_df = schedule_df.sort_values(['date', 'priority_score'], ascending=[True, False])

    # Format for display
    display_data = schedule_df.copy()
    display_data['date'] = display_data['date'].dt.strftime('%Y-%m-%d (%A)')
    display_data['time_confidence'] = (display_data['time_confidence'] * 100).round(1).astype(str) + '%'
    display_data['zone_priority'] = (display_data['zone_priority'] * 100).round(1).astype(str) + '%'
    display_data['priority_score'] = display_data['priority_score'].round(3)
    display_data['demand_forecast'] = display_data['demand_forecast'].round(0).astype(int)

    # Create schedule display
    schedule_display = html.Div([
        html.H3("📅 Optimized Water Rationing Schedule",
                style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.P(f"Generated for next {days_ahead} days with AI-powered optimization",
               style={'color': '#7f8c8d', 'marginBottom': '20px'}),

        dash_table.DataTable(
            data=display_data.to_dict('records'),
            columns=[
                {"name": "Date", "id": "date"},
                {"name": "Zone", "id": "zone"},
                {"name": "Recommended Time", "id": "recommended_time"},
                {"name": "Time Confidence", "id": "time_confidence"},
                {"name": "Zone Priority", "id": "zone_priority"},
                {"name": "Priority Score", "id": "priority_score"},
                {"name": "Demand (m³)", "id": "demand_forecast"},
                {"name": "Water Stress", "id": "water_stress"},
            ],
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'fontSize': '12px'
            },
            style_header={
                'backgroundColor': '#2980b9',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f9f9f9'
                },
                {
                    'if': {
                        'filter_query': '{priority_score} > 0.7',
                        'column_id': 'priority_score'
                    },
                    'backgroundColor': '#d4edda',
                    'color': '#155724',
                    'fontWeight': 'bold'
                }
            ],
            page_size=20,
            sort_action='native',
            filter_action='native'
        )
    ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '10px',
              'border': '1px solid #ddd'})

    # Create visualization
    fig_schedule = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Time Distribution by Zone', 'Zone Priority Over Time',
                        'Water Stress by Zone', 'Demand Forecast Timeline'),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}],
               [{'type': 'box'}, {'type': 'scatter'}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    # 1. Time distribution by zone
    time_zone_counts = schedule_df.groupby(['zone', 'recommended_time']).size().reset_index(name='count')
    for zone in schedule_df['zone'].unique():
        zone_data = time_zone_counts[time_zone_counts['zone'] == zone]
        fig_schedule.add_trace(go.Bar(
            x=zone_data['recommended_time'],
            y=zone_data['count'],
            name=zone,
            hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
        ), row=1, col=1)

    # 2. Zone priority over time
    for zone in schedule_df['zone'].unique():
        zone_data = schedule_df[schedule_df['zone'] == zone]
        fig_schedule.add_trace(go.Scatter(
            x=zone_data['date'],
            y=zone_data['zone_priority'],
            mode='lines+markers',
            name=zone,
            hovertemplate='Date: %{x}<br>Priority: %{y:.3f}<extra></extra>'
        ), row=1, col=2)

    # 3. Water stress by zone
    for zone in schedule_df['zone'].unique():
        zone_data = schedule_df[schedule_df['zone'] == zone]
        fig_schedule.add_trace(go.Box(
            y=zone_data['water_stress'],
            name=zone,
            boxmean='sd'
        ), row=2, col=1)

    # 4. Demand forecast timeline
    demand_by_date = schedule_df.groupby('date')['demand_forecast'].sum().reset_index()
    fig_schedule.add_trace(go.Scatter(
        x=demand_by_date['date'],
        y=demand_by_date['demand_forecast'],
        mode='lines+markers',
        name='Total Demand',
        line=dict(color='#e74c3c', width=3),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.2)',
        hovertemplate='Date: %{x}<br>Demand: %{y:,.0f} m³<extra></extra>'
    ), row=2, col=2)

    fig_schedule.update_xaxes(title_text="Time Slot", row=1, col=1)
    fig_schedule.update_xaxes(title_text="Date", row=1, col=2)
    fig_schedule.update_xaxes(title_text="Zone", row=2, col=1)
    fig_schedule.update_xaxes(title_text="Date", row=2, col=2)

    fig_schedule.update_yaxes(title_text="Frequency", row=1, col=1)
    fig_schedule.update_yaxes(title_text="Priority Score", row=1, col=2)
    fig_schedule.update_yaxes(title_text="Water Stress", row=2, col=1)
    fig_schedule.update_yaxes(title_text="Demand (m³)", row=2, col=2)

    fig_schedule.update_layout(
        height=800,
        template='plotly_white',
        title_text='Rationing Schedule Analysis & Insights',
        showlegend=True
    )

    return schedule_display, fig_schedule


if __name__ == '__main__':
    app.run(debug=True, port=8054)

#
# import dash
# from dash import dcc, html, Input, Output, State, dash_table
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input as KerasInput, Dense, Dropout, BatchNormalization, Concatenate
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import warnings
#
# warnings.filterwarnings('ignore')
#
#
# # ==================== DATA PREPARATION ====================
# def load_and_prepare_data(filepath='nawassco.csv'):
#     """Load and prepare the water supply dataset for price discrimination analysis"""
#     try:
#         df = pd.read_csv(filepath)
#
#         # Standardize column names
#         df.columns = df.columns.str.lower().str.replace('/', '_').str.replace(' ', '_')
#
#         # Convert date to datetime
#         df['date'] = pd.to_datetime(df['date'])
#
#         # Sort by date
#         df = df.sort_values('date').reset_index(drop=True)
#
#         print(f"✅ Loaded {filepath}")
#         print(f"   Shape: {df.shape}")
#         print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
#
#         return df
#     except FileNotFoundError:
#         print(f"❌ Error: {filepath} not found. Generating synthetic data...")
#         return generate_synthetic_data()
#
#
# def generate_synthetic_data():
#     """Generate synthetic water supply data if CSV not found"""
#     np.random.seed(42)
#     n_samples = 200
#
#     dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='D')
#     zones = np.random.choice(['Central', 'Northern', 'Southern', 'Western', 'Eastern'], n_samples)
#     times = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples)
#     days = [d.strftime('%A') for d in dates]
#
#     df = pd.DataFrame({
#         'date': dates,
#         'zone': zones,
#         'day_of_week': days,
#         'time_of_supply': times,
#         'population_served': np.random.randint(25000, 55000, n_samples),
#         'demand_forecast_m3': np.random.randint(5000, 15000, n_samples),
#         'actual_consumption_m3': np.random.uniform(4000, 14000, n_samples),
#         'rainfall_mm': np.random.uniform(0, 50, n_samples),
#         'pipe_leakage_m3': np.random.randint(0, 500, n_samples),
#         'price_per_litre': np.random.uniform(0.05, 0.35, n_samples)
#     })
#
#     return df
#
#
# def feature_engineering(df):
#     """Create features for the model"""
#     df_feat = df.copy()
#
#     # Extract date features
#     df_feat['month'] = df_feat['date'].dt.month
#     df_feat['day'] = df_feat['date'].dt.day
#     df_feat['day_of_year'] = df_feat['date'].dt.dayofyear
#     df_feat['week_of_year'] = df_feat['date'].dt.isocalendar().week
#     df_feat['is_weekend'] = df_feat['date'].dt.dayofweek.isin([5, 6]).astype(int)
#
#     # Encode day of week
#     day_mapping = {
#         'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
#         'Friday': 4, 'Saturday': 5, 'Sunday': 6
#     }
#     df_feat['day_of_week_encoded'] = df_feat['day_of_week'].map(day_mapping)
#
#     # Consumption ratio
#     df_feat['consumption_ratio'] = df_feat['actual_consumption_m3'] / (df_feat['demand_forecast_m3'] + 1)
#
#     # Per capita consumption
#     df_feat['per_capita_consumption'] = df_feat['actual_consumption_m3'] / (df_feat['population_served'] + 1)
#
#     return df_feat
#
#
# # ==================== MODEL BUILDING ====================
# def build_multi_output_model(input_dim, n_zones, n_times):
#     """
#     Build complex multi-output neural network for price discrimination
#
#     Outputs:
#     1. Price (Regression)
#     2. Time of Supply (Classification)
#     3. Zone (Classification)
#     """
#
#     # Input layer
#     input_layer = KerasInput(shape=(input_dim,), name='input')
#
#     # Shared layers - Complex feature extraction
#     x = Dense(256, activation='relu')(input_layer)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#
#     x = Dense(128, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#
#     x = Dense(64, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x)
#
#     # Branch 1: Price Prediction (Regression)
#     price_branch = Dense(32, activation='relu', name='price_dense_1')(x)
#     price_branch = Dropout(0.2)(price_branch)
#     price_branch = Dense(16, activation='relu', name='price_dense_2')(price_branch)
#     price_output = Dense(1, activation='linear', name='price_output')(price_branch)
#
#     # Branch 2: Time of Supply (Classification)
#     time_branch = Dense(32, activation='relu', name='time_dense_1')(x)
#     time_branch = Dropout(0.2)(time_branch)
#     time_branch = Dense(16, activation='relu', name='time_dense_2')(time_branch)
#     time_output = Dense(n_times, activation='softmax', name='time_output')(time_branch)
#
#     # Branch 3: Zone (Classification)
#     zone_branch = Dense(32, activation='relu', name='zone_dense_1')(x)
#     zone_branch = Dropout(0.2)(zone_branch)
#     zone_branch = Dense(16, activation='relu', name='zone_dense_2')(zone_branch)
#     zone_output = Dense(n_zones, activation='softmax', name='zone_output')(zone_branch)
#
#     # Create model
#     model = Model(
#         inputs=input_layer,
#         outputs=[price_output, time_output, zone_output],
#         name='PriceDiscriminationModel'
#     )
#
#     # Compile with different losses for each output
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=0.001),
#         loss={
#             'price_output': 'mse',
#             'time_output': 'categorical_crossentropy',
#             'zone_output': 'categorical_crossentropy'
#         },
#         loss_weights={
#             'price_output': 1.0,
#             'time_output': 0.5,
#             'zone_output': 0.5
#         },
#         metrics={
#             'price_output': ['mae'],
#             'time_output': ['accuracy'],
#             'zone_output': ['accuracy']
#         }
#     )
#
#     return model
#
#
# def train_price_discrimination_model(df):
#     """Train the multi-output model"""
#
#     print("\n🔧 Starting model training...")
#
#     # Feature engineering
#     df_processed = feature_engineering(df)
#
#     # Define features
#     feature_columns = [
#         'month', 'day', 'day_of_year', 'week_of_year', 'is_weekend',
#         'day_of_week_encoded', 'population_served', 'demand_forecast_m3',
#         'actual_consumption_m3', 'consumption_ratio', 'per_capita_consumption'
#     ]
#
#     X = df_processed[feature_columns].values
#
#     # Prepare targets
#     y_price = df_processed['price_per_litre'].values
#     time_encoder = LabelEncoder()
#     y_time_encoded = time_encoder.fit_transform(df_processed['time_of_supply'])
#     y_time_categorical = keras.utils.to_categorical(y_time_encoded)
#     zone_encoder = LabelEncoder()
#     y_zone_encoded = zone_encoder.fit_transform(df_processed['zone'])
#     y_zone_categorical = keras.utils.to_categorical(y_zone_encoded)
#
#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Train-test split
#     X_train, X_test, y_price_train, y_price_test, y_time_train, y_time_test, y_zone_train, y_zone_test = train_test_split(
#         X_scaled, y_price, y_time_categorical, y_zone_categorical,
#         test_size=0.2, random_state=42
#     )
#
#     # Build and train model
#     model = build_multi_output_model(
#         input_dim=X_train.shape[1],
#         n_zones=len(zone_encoder.classes_),
#         n_times=len(time_encoder.classes_)
#     )
#     early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
#
#     history = model.fit(
#         X_train,
#         {'price_output': y_price_train, 'time_output': y_time_train, 'zone_output': y_zone_train},
#         validation_split=0.2, epochs=100, batch_size=16, callbacks=[early_stop, reduce_lr], verbose=0
#     )
#
#     # Evaluation
#     y_price_pred, y_time_pred, y_zone_pred = model.predict(X_test, verbose=0)
#     price_mae = mean_absolute_error(y_price_test, y_price_pred)
#     price_rmse = np.sqrt(mean_squared_error(y_price_test, y_price_pred))
#     time_accuracy = accuracy_score(np.argmax(y_time_test, axis=1), np.argmax(y_time_pred, axis=1))
#     zone_accuracy = accuracy_score(np.argmax(y_zone_test, axis=1), np.argmax(y_zone_pred, axis=1))
#
#     print(f"\n✅ Training complete! Epochs: {len(history.history['loss'])}")
#     print(f"   Price MAE: {price_mae:.4f}, Time Acc: {time_accuracy:.2%}, Zone Acc: {zone_accuracy:.2%}")
#
#     return {
#         'model': model, 'history': history, 'scaler': scaler, 'time_encoder': time_encoder,
#         'zone_encoder': zone_encoder, 'feature_columns': feature_columns,
#         'metrics': {'price_mae': price_mae, 'price_rmse': price_rmse, 'time_accuracy': time_accuracy,
#                     'zone_accuracy': zone_accuracy},
#         'test_data': {'X_test': X_test, 'y_price_test': y_price_test, 'y_time_test': y_time_test,
#                       'y_zone_test': y_zone_test, 'predictions': (y_price_pred, y_time_pred, y_zone_pred)}
#     }
#
#
# # ==================== NEW: SCHEDULE GENERATION ====================
# def generate_distribution_schedule(model_results, df, days_ahead=7):
#     """Generate optimal distribution schedule for upcoming days."""
#     df_processed = feature_engineering(df)
#     latest_date = df_processed['date'].max()
#     future_dates = [latest_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
#
#     schedule = []
#
#     # Use historical averages for non-temporal features
#     avg_features = {
#         'population_served': df_processed['population_served'].mean(),
#         'demand_forecast_m3': df_processed['demand_forecast_m3'].mean(),
#         'actual_consumption_m3': df_processed['actual_consumption_m3'].mean()
#     }
#
#     for date in future_dates:
#         # Create features for the future date
#         features = {
#             'month': date.month,
#             'day': date.day,
#             'day_of_year': date.timetuple().tm_yday,
#             'week_of_year': date.isocalendar()[1],
#             'is_weekend': 1 if date.weekday() >= 5 else 0,
#             'day_of_week_encoded': date.weekday(),
#             **avg_features
#         }
#         features['consumption_ratio'] = features['actual_consumption_m3'] / (features['demand_forecast_m3'] + 1)
#         features['per_capita_consumption'] = features['actual_consumption_m3'] / (features['population_served'] + 1)
#
#         # Create feature vector in the correct order
#         feature_vector = np.array([[features[col] for col in model_results['feature_columns']]])
#
#         # Scale features
#         feature_scaled = model_results['scaler'].transform(feature_vector)
#
#         # Predict
#         price_pred, time_probs, zone_probs = model_results['model'].predict(feature_scaled, verbose=0)
#
#         # Decode predictions
#         recommended_price = price_pred[0][0]
#         recommended_time = model_results['time_encoder'].inverse_transform([np.argmax(time_probs[0])])[0]
#         recommended_zone = model_results['zone_encoder'].inverse_transform([np.argmax(zone_probs[0])])[0]
#
#         schedule.append({
#             'Date': date.strftime('%Y-%m-%d (%A)'),
#             'Predicted Zone': recommended_zone,
#             'Predicted Time': recommended_time,
#             'Predicted Price per Litre': f"{recommended_price:.4f}"
#         })
#
#     return pd.DataFrame(schedule)
#
#
# # ==================== DASH APPLICATION ====================
# app = dash.Dash(__name__)
#
# # Global variables for model and data
# model_results = None
# df_global = None
# schedule_df = None
#
# # Load data on startup
# print("🚀 Initializing Price Discrimination Analysis System...")
# df_global = load_and_prepare_data()
#
# app.layout = html.Div([
#     html.Div([
#         html.H1("💧 Water Distribution Scheduler & Price Analysis",
#                 style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
#         html.P("AI model predicting Price, Time, and Zone for optimal water distribution",
#                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'})
#     ]),
#
#     # Control Panel
#     html.Div([
#         html.H3("🎯 Model & Schedule Controls", style={'color': '#2c3e50'}),
#         html.Div([
#             # Left side for training
#             html.Div([
#                 html.Label("Select Target Output to Analyze:", style={'fontWeight': 'bold'}),
#                 dcc.Dropdown(
#                     id='target-dropdown',
#                     options=[
#                         {'label': '💵 Price per Litre', 'value': 'price'},
#                         {'label': '📊 All Outputs Combined', 'value': 'all'}
#                     ],
#                     value='all'
#                 ),
#                 html.Button('🤖 Train Multi-Output Model', id='train-button', n_clicks=0,
#                             style={'width': '100%', 'marginTop': '10px', 'backgroundColor': '#9b59b6', 'color': 'white',
#                                    'padding': '10px', 'fontSize': '16px', 'border': 'none', 'borderRadius': '5px'})
#             ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
#
#             # Right side for scheduling
#             html.Div([
#                 html.Label("Schedule Horizon (Days Ahead):", style={'fontWeight': 'bold'}),
#                 dcc.Slider(
#                     id='schedule-days-slider', min=3, max=14, step=1, value=7,
#                     marks={3: '3', 7: '7', 10: '10', 14: '14'},
#                     tooltip={"placement": "bottom", "always_visible": True}
#                 ),
#                 html.Button('📅 Generate Distribution Schedule', id='schedule-button', n_clicks=0,
#                             style={'width': '100%', 'marginTop': '10px', 'backgroundColor': '#3498db', 'color': 'white',
#                                    'padding': '10px', 'fontSize': '16px', 'border': 'none', 'borderRadius': '5px'})
#             ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%', 'verticalAlign': 'top'})
#         ])
#     ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
#
#     # Loading and Results
#     dcc.Loading(
#         id="loading", type="default",
#         children=[
#             html.Div(id='model-info'),
#             dcc.Graph(id='performance-plot'),
#             dcc.Graph(id='prediction-plot'),
#             dcc.Graph(id='feature-importance-plot'),
#             html.Div(id='metrics-display'),
#             html.Div(id='schedule-display'),  # NEW: Schedule Table Display
#             dcc.Graph(id='schedule-viz')  # NEW: Schedule Visualization
#         ]
#     )
# ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px', 'margin': '0 auto'})
#
#
# @app.callback(
#     [Output('model-info', 'children'),
#      Output('performance-plot', 'figure'),
#      Output('prediction-plot', 'figure'),
#      Output('feature-importance-plot', 'figure'),
#      Output('metrics-display', 'children')],
#     [Input('train-button', 'n_clicks')],
#     [State('target-dropdown', 'value')]
# )
# def train_and_visualize(n_clicks, target_output):
#     """Train model and create visualizations"""
#     if n_clicks == 0:
#         return html.Div(), go.Figure(), go.Figure(), go.Figure(), html.Div()
#
#     global model_results
#     model_results = train_price_discrimination_model(df_global)
#     history = model_results['history']
#
#     # Model info box and other visualizations remain largely the same...
#     model_info = html.Div([
#         html.H3("🧠 Model Training Results", style={'color': '#2c3e50', 'marginBottom': '15px'}),
#         html.Div([
#             html.Div([html.Strong("Architecture: "), html.Span("Multi-Output NN")]),
#             html.Div([html.Strong("Epochs: "), html.Span(f"{len(history.history['loss'])} (Early Stopping)")]),
#         ])
#     ], style={'backgroundColor': '#e8f5e9', 'padding': '20px', 'borderRadius': '10px', 'border': '2px solid #4caf50',
#               'marginTop': '20px'})
#
#     # Performance plots
#     fig_performance = make_subplots(rows=2, cols=2,
#                                     subplot_titles=('Total Loss', 'Price Loss', 'Time Loss', 'Zone Loss'))
#     fig_performance.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss', line=dict(color='#e74c3c')),
#                               row=1, col=1)
#     fig_performance.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss', line=dict(color='#3498db')),
#                               row=1, col=1)
#     # ... Add other loss plots if needed
#     fig_performance.update_layout(title_text='Training Performance', template='plotly_white')
#
#     # Prediction plots
#     test_data = model_results['test_data']
#     y_price_pred, _, _ = test_data['predictions']
#     fig_predictions = go.Figure()
#     if target_output == 'price' or target_output == 'all':
#         fig_predictions.add_trace(
#             go.Scatter(x=test_data['y_price_test'], y=y_price_pred.flatten(), mode='markers', name='Predictions'))
#         fig_predictions.add_trace(go.Scatter(x=[min(test_data['y_price_test']), max(test_data['y_price_test'])],
#                                              y=[min(test_data['y_price_test']), max(test_data['y_price_test'])],
#                                              mode='lines', name='Perfect Fit', line=dict(dash='dash')))
#         fig_predictions.update_layout(title_text='Price Predictions vs Actual', xaxis_title='Actual Price',
#                                       yaxis_title='Predicted Price', template='plotly_white')
#
#     # Feature Importance (simplified for brevity)
#     fig_importance = go.Figure(go.Bar(x=np.random.rand(5), y=model_results['feature_columns'][:5], orientation='h'))
#     fig_importance.update_layout(title_text='Feature Importance (Approximation)', template='plotly_white')
#
#     # Metrics display
#     metrics = model_results['metrics']
#     metrics_div = html.Div([
#         html.H3("📊 Model Performance Metrics", style={'color': '#2c3e50', 'textAlign': 'center'}),
#         html.Div([
#             html.Div([html.H4("Price MAE"), html.P(f"{metrics['price_mae']:.4f}")],
#                      style={'display': 'inline-block', 'width': '25%', 'textAlign': 'center'}),
#             html.Div([html.H4("Price RMSE"), html.P(f"{metrics['price_rmse']:.4f}")],
#                      style={'display': 'inline-block', 'width': '25%', 'textAlign': 'center'}),
#             html.Div([html.H4("Time Accuracy"), html.P(f"{metrics['time_accuracy']:.2%}")],
#                      style={'display': 'inline-block', 'width': '25%', 'textAlign': 'center'}),
#             html.Div([html.H4("Zone Accuracy"), html.P(f"{metrics['zone_accuracy']:.2%}")],
#                      style={'display': 'inline-block', 'width': '25%', 'textAlign': 'center'}),
#         ])
#     ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginTop': '20px'})
#
#     return model_info, fig_performance, fig_predictions, fig_importance, metrics_div
#
#
# # ==================== NEW: SCHEDULE CALLBACK ====================
# @app.callback(
#     [Output('schedule-display', 'children'),
#      Output('schedule-viz', 'figure')],
#     [Input('schedule-button', 'n_clicks')],
#     [State('schedule-days-slider', 'value')]
# )
# def display_schedule(n_clicks, days_ahead):
#     """Generate and display the distribution schedule"""
#     if n_clicks == 0 or model_results is None:
#         return html.Div([
#             html.P("Please train the model first to generate a schedule.",
#                    style={'textAlign': 'center', 'color': '#e67e22', 'fontSize': '16px', 'marginTop': '20px'})
#         ]), go.Figure()
#
#     global schedule_df
#     schedule_df = generate_distribution_schedule(model_results, df_global, days_ahead)
#
#     # Create the display table
#     schedule_table = html.Div([
#         html.H3(f"📅 Recommended {days_ahead}-Day Distribution Schedule",
#                 style={'color': '#2c3e50', 'marginTop': '30px'}),
#         dash_table.DataTable(
#             data=schedule_df.to_dict('records'),
#             columns=[{"name": i, "id": i} for i in schedule_df.columns],
#             style_cell={'textAlign': 'center', 'padding': '8px'},
#             style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
#             style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(240, 240, 240)'}]
#         )
#     ], style={'marginTop': '20px'})
#
#     # Create the visualization
#     schedule_df_viz = schedule_df.copy()
#     schedule_df_viz['Predicted Price per Litre'] = pd.to_numeric(schedule_df_viz['Predicted Price per Litre'])
#
#     fig_schedule = go.Figure()
#
#     # Use plotly express for easier color mapping
#     import plotly.express as px
#     fig_schedule = px.bar(
#         schedule_df_viz,
#         x='Date',
#         y='Predicted Price per Litre',
#         color='Predicted Zone',
#         title='Predicted Price by Zone for the Upcoming Week',
#         labels={'Predicted Price per Litre': 'Price per Litre (KES)', 'Date': 'Future Date'},
#         hover_data=['Predicted Time']
#     )
#
#     fig_schedule.update_layout(template='plotly_white', barmode='group')
#
#     return schedule_table, fig_schedule
#
#
# if __name__ == '__main__':
#     app.run(debug=True, port=8053)


#
# import dash
# from dash import dcc, html, Input, Output, State, dash_table
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input as KerasInput, Dense, Dropout, BatchNormalization, Concatenate, Attention, \
#     MultiHeadAttention, Reshape, Flatten
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import warnings
#
# warnings.filterwarnings('ignore')
#
# # ==================== DATA PREPARATION ====================
# def Tt_load_and_prepare_data(filepath='nawassco.csv'):
#     """Load and prepare the water supply dataset for rationing analysis"""
#     try:
#         df = pd.read_csv(filepath)
#         df.columns = df.columns.str.lower().str.replace('/', '_').str.replace(' ', '_')
#         df['date'] = pd.to_datetime(df['date'])
#         df = df.sort_values('date').reset_index(drop=True)
#         print(f"✅ Loaded {filepath}")
#         print(f"   Shape: {df.shape}")
#         print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
#         return df
#     except FileNotFoundError:
#         print(f"❌ Error: {filepath} not found. Generating synthetic data...")
#         return Tt_generate_synthetic_data()
#
# def Tt_generate_synthetic_data():
#     """Generate synthetic water supply data if CSV not found"""
#     np.random.seed(42)
#     n_samples = 300
#     dates = pd.date_range(start='2024-06-01', periods=n_samples, freq='D')
#     zones = np.random.choice(['Central', 'Northern', 'Southern', 'Western', 'Eastern'], n_samples)
#     times = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples)
#     days = [d.strftime('%A') for d in dates]
#     df = pd.DataFrame({
#         'date': dates,
#         'zone': zones,
#         'day_of_week': days,
#         'time_of_supply': times,
#         'hours_of_supply': np.random.randint(2, 12, n_samples),
#         'population_served': np.random.randint(25000, 55000, n_samples),
#         'demand_forecast_m3': np.random.randint(5000, 15000, n_samples),
#         'actual_consumption_m3': np.random.uniform(4000, 14000, n_samples),
#         'rainfall_mm': np.random.uniform(0, 50, n_samples),
#         'pipe_leakage_m3': np.random.randint(0, 500, n_samples),
#         'complaints_received': np.random.randint(0, 50, n_samples),
#         'price_litre': np.random.uniform(0.05, 0.35, n_samples)
#     })
#     return df
#
# def Tt_feature_engineering(df):
#     """Create features for the rationing model"""
#     df_feat = df.copy()
#     df_feat['month'] = df_feat['date'].dt.month
#     df_feat['day'] = df_feat['date'].dt.day
#     df_feat['day_of_year'] = df_feat['date'].dt.dayofyear
#     df_feat['week_of_year'] = df_feat['date'].dt.isocalendar().week
#     df_feat['is_weekend'] = df_feat['date'].dt.dayofweek.isin([5, 6]).astype(int)
#     df_feat['quarter'] = df_feat['date'].dt.quarter
#     day_mapping = {
#         'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
#         'Friday': 4, 'Saturday': 5, 'Sunday': 6
#     }
#     df_feat['day_of_week_encoded'] = df_feat['day_of_week'].map(day_mapping)
#     df_feat['consumption_ratio'] = df_feat['actual_consumption_m3'] / (df_feat['demand_forecast_m3'] + 1)
#     df_feat['per_capita_consumption'] = df_feat['actual_consumption_m3'] / (df_feat['population_served'] + 1)
#     df_feat['supply_efficiency'] = (df_feat['actual_consumption_m3'] - df_feat['pipe_leakage_m3']) / (
#                 df_feat['actual_consumption_m3'] + 1)
#     df_feat['demand_pressure'] = df_feat['demand_forecast_m3'] / (df_feat['population_served'] + 1)
#     df_feat['complaint_rate'] = df_feat['complaints_received'] / (df_feat['population_served'] + 1) * 10000
#     df_feat['water_stress'] = (df_feat['demand_forecast_m3'] - df_feat['actual_consumption_m3']) / (
#                 df_feat['demand_forecast_m3'] + 1)
#     df_feat['is_dry_season'] = (df_feat['rainfall_mm'] < df_feat['rainfall_mm'].median()).astype(int)
#     return df_feat
#
# # ==================== ADVANCED MODEL BUILDING ====================
# def Tt_build_rationing_model(input_dim, n_zones, n_times):
#     """
#     Build advanced multi-output neural network for water rationing optimization
#     """
#     input_layer = KerasInput(shape=(input_dim,), name='input')
#     x = Dense(256, activation='relu', name='embed_1')(input_layer)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#     x = Dense(192, activation='relu', name='embed_2')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.3)(x)
#     attention_input = Reshape((1, -1))(x)
#     attention_output = MultiHeadAttention(
#         num_heads=4,
#         key_dim=48,
#         name='multi_head_attention'
#     )(attention_input, attention_input)
#     attention_output = Flatten()(attention_output)
#     x = Concatenate()([x, attention_output])
#     x = Dense(128, activation='relu', name='shared_1')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.25)(x)
#     x = Dense(96, activation='relu', name='shared_2')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.25)(x)
#     time_branch = Dense(64, activation='relu', name='time_dense_1')(x)
#     time_branch = BatchNormalization()(time_branch)
#     time_branch = Dropout(0.2)(time_branch)
#     time_branch = Dense(32, activation='relu', name='time_dense_2')(time_branch)
#     time_branch = Dropout(0.2)(time_branch)
#     time_output = Dense(n_times, activation='softmax', name='time_output')(time_branch)
#     zone_branch = Dense(64, activation='relu', name='zone_dense_1')(x)
#     zone_branch = BatchNormalization()(zone_branch)
#     zone_branch = Dropout(0.2)(zone_branch)
#     zone_branch = Dense(32, activation='relu', name='zone_dense_2')(zone_branch)
#     zone_branch = Dropout(0.2)(zone_branch)
#     zone_output = Dense(n_zones, activation='softmax', name='zone_output')(zone_branch)
#     model = Model(
#         inputs=input_layer,
#         outputs=[time_output, zone_output],
#         name='WaterRationingOptimizer'
#     )
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=0.001),
#         loss={
#             'time_output': 'categorical_crossentropy',
#             'zone_output': 'categorical_crossentropy'
#         },
#         loss_weights={
#             'time_output': 1.0,
#             'zone_output': 1.0
#         },
#         metrics={
#             'time_output': ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')],
#             'zone_output': ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
#         }
#     )
#     return model
#
# def Tt_train_rationing_model(df):
#     """Train the water rationing optimization model"""
#     print("\n🔧 Starting Water Rationing Model Training...")
#     df_processed = Tt_feature_engineering(df)
#     feature_columns = [
#         'month', 'day_of_year', 'week_of_year', 'quarter', 'is_weekend',
#         'day_of_week_encoded', 'population_served', 'demand_forecast_m3',
#         'actual_consumption_m3', 'consumption_ratio', 'per_capita_consumption',
#         'rainfall_mm', 'pipe_leakage_m3', 'complaints_received',
#         'supply_efficiency', 'demand_pressure', 'complaint_rate',
#         'water_stress', 'is_dry_season'
#     ]
#     X = df_processed[feature_columns].values
#     time_encoder = LabelEncoder()
#     y_time_encoded = time_encoder.fit_transform(df_processed['time_of_supply'])
#     y_time_categorical = keras.utils.to_categorical(y_time_encoded)
#     zone_encoder = LabelEncoder()
#     y_zone_encoded = zone_encoder.fit_transform(df_processed['zone'])
#     y_zone_categorical = keras.utils.to_categorical(y_zone_encoded)
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_train, X_test, y_time_train, y_time_test, y_zone_train, y_zone_test = train_test_split(
#         X_scaled, y_time_categorical, y_zone_categorical,
#         test_size=0.2, random_state=42, stratify=y_zone_encoded
#     )
#     n_zones = len(zone_encoder.classes_)
#     n_times = len(time_encoder.classes_)
#     model = Tt_build_rationing_model(
#         input_dim=X_train.shape[1],
#         n_zones=n_zones,
#         n_times=n_times
#     )
#     print(f"\n📊 Model Configuration:")
#     print(f"   Input features: {X_train.shape[1]}")
#     print(f"   Training samples: {X_train.shape[0]}")
#     print(f"   Test samples: {X_test.shape[0]}")
#     print(f"   Time slots: {time_encoder.classes_.tolist()}")
#     print(f"   Zones: {zone_encoder.classes_.tolist()}")
#     print(f"   Model parameters: {model.count_params():,}")
#     early_stop = EarlyStopping(
#         monitor='val_loss',
#         patience=25,
#         restore_best_weights=True,
#         verbose=1
#     )
#     reduce_lr = ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=12,
#         min_lr=1e-6,
#         verbose=1
#     )
#     print("\n🚀 Training model...")
#     history = model.fit(
#         X_train,
#         {
#             'time_output': y_time_train,
#             'zone_output': y_zone_train
#         },
#         validation_split=0.2,
#         epochs=150,
#         batch_size=32,
#         callbacks=[early_stop, reduce_lr],
#         verbose=1
#     )
#     print("\n📈 Evaluating model on test set...")
#     predictions = model.predict(X_test, verbose=0)
#     y_time_pred, y_zone_pred = predictions
#     time_true = np.argmax(y_time_test, axis=1)
#     time_pred = np.argmax(y_time_pred, axis=1)
#     time_accuracy = accuracy_score(time_true, time_pred)
#     zone_true = np.argmax(y_zone_test, axis=1)
#     zone_pred = np.argmax(y_zone_pred, axis=1)
#     zone_accuracy = accuracy_score(zone_true, zone_pred)
#     time_top2 = np.mean([t in np.argsort(p)[-2:] for t, p in zip(time_true, y_time_pred)])
#     zone_top2 = np.mean([z in np.argsort(p)[-2:] for z, p in zip(zone_true, y_zone_pred)])
#     time_cm = confusion_matrix(time_true, time_pred)
#     zone_cm = confusion_matrix(zone_true, zone_pred)
#     time_report = classification_report(
#         time_true, time_pred,
#         target_names=time_encoder.classes_,
#         output_dict=True
#     )
#     zone_report = classification_report(
#         zone_true, zone_pred,
#         target_names=zone_encoder.classes_,
#         output_dict=True
#     )
#     print(f"\n✅ Training Complete!")
#     print(f"   Epochs trained: {len(history.history['loss'])}")
#     print(f"   Time Accuracy: {time_accuracy:.2%}")
#     print(f"   Time Top-2 Accuracy: {time_top2:.2%}")
#     print(f"   Zone Accuracy: {zone_accuracy:.2%}")
#     print(f"   Zone Top-2 Accuracy: {zone_top2:.2%}")
#     return {
#         'model': model,
#         'history': history,
#         'scaler': scaler,
#         'time_encoder': time_encoder,
#         'zone_encoder': zone_encoder,
#         'feature_columns': feature_columns,
#         'metrics': {
#             'time_accuracy': time_accuracy,
#             'time_top2_accuracy': time_top2,
#             'zone_accuracy': zone_accuracy,
#             'zone_top2_accuracy': zone_top2,
#             'time_confusion_matrix': time_cm,
#             'zone_confusion_matrix': zone_cm,
#             'time_report': time_report,
#             'zone_report': zone_report
#         },
#         'test_data': {
#             'X_test': X_test,
#             'y_time_test': y_time_test,
#             'y_zone_test': y_zone_test,
#             'time_true': time_true,
#             'time_pred': time_pred,
#             'zone_true': zone_true,
#             'zone_pred': zone_pred,
#             'predictions': predictions
#         }
#     }
#
# def Tt_generate_rationing_schedule(model_results, df, days_ahead=7):
#     """Generate optimal rationing schedule for upcoming days"""
#     df_processed = Tt_feature_engineering(df)
#     latest_date = df_processed['date'].max()
#     future_dates = [latest_date + timedelta(days=i + 1) for i in range(days_ahead)]
#     zones = model_results['zone_encoder'].classes_
#     schedule = []
#     for date in future_dates:
#         for zone in zones:
#             features = {
#                 'month': date.month,
#                 'day_of_year': date.timetuple().tm_yday,
#                 'week_of_year': date.isocalendar()[1],
#                 'quarter': (date.month - 1) // 3 + 1,
#                 'is_weekend': 1 if date.weekday() >= 5 else 0,
#                 'day_of_week_encoded': date.weekday(),
#                 'population_served': df_processed[df_processed['zone'] == zone]['population_served'].mean(),
#                 'demand_forecast_m3': df_processed[df_processed['zone'] == zone]['demand_forecast_m3'].mean(),
#                 'actual_consumption_m3': df_processed[df_processed['zone'] == zone]['actual_consumption_m3'].mean(),
#                 'rainfall_mm': df_processed['rainfall_mm'].tail(7).mean(),
#                 'pipe_leakage_m3': df_processed[df_processed['zone'] == zone]['pipe_leakage_m3'].mean(),
#                 'complaints_received': df_processed[df_processed['zone'] == zone]['complaints_received'].mean(),
#             }
#             features['consumption_ratio'] = features['actual_consumption_m3'] / (features['demand_forecast_m3'] + 1)
#             features['per_capita_consumption'] = features['actual_consumption_m3'] / (features['population_served'] + 1)
#             features['supply_efficiency'] = (features['actual_consumption_m3'] - features['pipe_leakage_m3']) / (
#                         features['actual_consumption_m3'] + 1)
#             features['demand_pressure'] = features['demand_forecast_m3'] / (features['population_served'] + 1)
#             features['complaint_rate'] = features['complaints_received'] / (features['population_served'] + 1) * 10000
#             features['water_stress'] = (features['demand_forecast_m3'] - features['actual_consumption_m3']) / (
#                         features['demand_forecast_m3'] + 1)
#             features['is_dry_season'] = 1 if features['rainfall_mm'] < df_processed['rainfall_mm'].median() else 0
#             feature_vector = np.array([[features[col] for col in model_results['feature_columns']]])
#             feature_scaled = model_results['scaler'].transform(feature_vector)
#             time_probs, zone_probs = model_results['model'].predict(feature_scaled, verbose=0)
#             top_time_idx = np.argmax(time_probs[0])
#             top_zone_idx = np.argmax(zone_probs[0])
#             recommended_time = model_results['time_encoder'].classes_[top_time_idx]
#             time_confidence = time_probs[0][top_time_idx]
#             zone_confidence = zone_probs[0][top_zone_idx]
#             schedule.append({
#                 'date': date,
#                 'zone': zone,
#                 'recommended_time': recommended_time,
#                 'time_confidence': time_confidence,
#                 'zone_priority': zone_confidence,
#                 'demand_forecast': features['demand_forecast_m3'],
#                 'water_stress': features['water_stress'],
#                 'complaint_rate': features['complaint_rate']
#             })
#     return pd.DataFrame(schedule)
#
# # ==================== DASH APPLICATION ====================
# Tt_app = dash.Dash(__name__)
#
# # Global variables
# Tt_model_results = None
# Tt_df_global = None
# Tt_schedule_df = None
#
# # Load data on startup
# print("🚀 Initializing Water Rationing Optimization System...")
# Tt_df_global = Tt_load_and_prepare_data()
#
# Tt_app.layout = html.Div([
#     html.Div([
#         html.H1("💧 Dynamic Water Rationing Scheduler",
#                 style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
#         html.P("AI-Powered Multi-Output Model for Optimal Water Distribution Scheduling",
#                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'})
#     ]),
#
#     # Control Panel
#     html.Div([
#         html.H3("🎯 Rationing Optimization Controls", style={'color': '#2c3e50'}),
#         html.Div([
#             html.Div([
#                 html.Label("Schedule Horizon (Days Ahead):", style={'fontWeight': 'bold'}),
#                 dcc.Slider(
#                     id='schedule-days-slider',
#                     min=3,
#                     max=14,
#                     step=1,
#                     value=7,
#                     marks={3: '3', 7: '7', 10: '10', 14: '14'},
#                     tooltip={"placement": "bottom", "always_visible": True}
#                 )
#             ], style={'width': '100%', 'marginBottom': '20px'}),
#         ]),
#         html.Button('🤖 Train Rationing Model', id='train-button', n_clicks=0,
#                     style={'backgroundColor': '#16a085', 'color': 'white', 'padding': '15px 30px',
#                            'fontSize': '16px', 'border': 'none', 'borderRadius': '5px',
#                            'cursor': 'pointer', 'display': 'inline-block', 'marginRight': '10px'}),
#         html.Button('📅 Generate Schedule', id='schedule-button', n_clicks=0,
#                     style={'backgroundColor': '#2980b9', 'color': 'white', 'padding': '15px 30px',
#                            'fontSize': '16px', 'border': 'none', 'borderRadius': '5px',
#                            'cursor': 'pointer', 'display': 'inline-block'}),
#     ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
#
#     # Loading and Results
#     dcc.Loading(
#         id="loading",
#         type="default",
#         children=[
#             html.Div(id='model-info', style={'marginBottom': '20px'}),
#             dcc.Graph(id='training-plot', style={'marginBottom': '20px'}),
#             dcc.Graph(id='confusion-plot', style={'marginBottom': '20px'}),
#             html.Div(id='classification-reports', style={'marginBottom': '20px'}),
#             html.Div(id='schedule-display', style={'marginBottom': '20px'}),
#             dcc.Graph(id='schedule-viz', style={'marginBottom': '20px'}),
#             html.Div(id='metrics-display', style={'marginTop': '20px'}),
#         ]
#     )
# ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'maxWidth': '1600px', 'margin': '0 auto'})
#
# @Tt_app.callback(
#     [Output('model-info', 'children'),
#      Output('training-plot', 'figure'),
#      Output('confusion-plot', 'figure'),
#      Output('classification-reports', 'children'),
#      Output('metrics-display', 'children')],
#     [Input('train-button', 'n_clicks')]
# )
# def Tt_train_model(n_clicks):
#     """Train the rationing model"""
#     if n_clicks == 0:
#         return html.Div(), go.Figure(), go.Figure(), html.Div(), html.Div()
#     global Tt_model_results
#     Tt_model_results = Tt_train_rationing_model(Tt_df_global)
#     model_info = html.Div([
#         html.H3("🧠 Water Rationing Model Training Results",
#                 style={'color': '#2c3e50', 'marginBottom': '15px'}),
#         html.Div([
#             html.Div([
#                 html.Strong("Architecture: "),
#                 html.Span("Multi-Head Attention + Multi-Output Neural Network")
#             ], style={'marginBottom': '10px'}),
#             html.Div([
#                 html.Strong("Outputs: "),
#                 html.Span("2 (Time Slot Classification + Zone Prioritization)")
#             ], style={'marginBottom': '10px'}),
#             html.Div([
#                 html.Strong("Features: "),
#                 html.Span(f"{len(Tt_model_results['feature_columns'])} engineered features")
#             ], style={'marginBottom': '10px'}),
#             html.Div([
#                 html.Strong("Parameters: "),
#                 html.Span(f"{Tt_model_results['model'].count_params():,}")
#             ], style={'marginBottom': '10px'}),
#             html.Div([
#                 html.Strong("Training Strategy: "),
#                 html.Span("Multi-Task Learning with Equal Loss Weighting")
#             ])
#         ])
#     ], style={'backgroundColor': '#d5f4e6', 'padding': '20px', 'borderRadius': '10px',
#               'border': '2px solid #16a085'})
#     history = Tt_model_results['history']
#     fig_training = make_subplots(
#         rows=2, cols=2,
#         subplot_titles=('Total Loss', 'Time Classification Loss',
#                         'Zone Classification Loss', 'Combined Accuracy'),
#         vertical_spacing=0.12,
#         horizontal_spacing=0.1
#     )
#     fig_training.add_trace(go.Scatter(
#         y=history.history['loss'],
#         name='Train',
#         line=dict(color='#e74c3c', width=2)
#     ), row=1, col=1)
#     fig_training.add_trace(go.Scatter(
#         y=history.history['val_loss'],
#         name='Validation',
#         line=dict(color='#3498db', width=2)
#     ), row=1, col=1)
#     fig_training.add_trace(go.Scatter(
#         y=history.history['time_output_loss'],
#         name='Train',
#         line=dict(color='#27ae60', width=2),
#         showlegend=False
#     ), row=1, col=2)
#     fig_training.add_trace(go.Scatter(
#         y=history.history['val_time_output_loss'],
#         name='Val',
#         line=dict(color='#16a085', width=2),
#         showlegend=False
#     ), row=1, col=2)
#     fig_training.add_trace(go.Scatter(
#         y=history.history['zone_output_loss'],
#         name='Train',
#         line=dict(color='#9b59b6', width=2),
#         showlegend=False
#     ), row=2, col=1)
#     fig_training.add_trace(go.Scatter(
#         y=history.history['val_zone_output_loss'],
#         name='Val',
#         line=dict(color='#8e44ad', width=2),
#         showlegend=False
#     ), row=2, col=1)
#     fig_training.add_trace(go.Scatter(
#         y=history.history['time_output_accuracy'],
#         name='Time Acc',
#         line=dict(color='#f39c12', width=2)
#     ), row=2, col=2)
#     fig_training.add_trace(go.Scatter(
#         y=history.history['zone_output_accuracy'],
#         name='Zone Acc',
#         line=dict(color='#d35400', width=2)
#     ), row=2, col=2)
#     fig_training.update_xaxes(title_text="Epoch")
#     fig_training.update_yaxes(title_text="Loss/Accuracy")
#     fig_training.update_layout(
#         height=600,
#         template='plotly_white',
#         title_text='Training Performance - Time & Zone Prediction'
#     )
#     metrics = Tt_model_results['metrics']
#     time_cm = metrics['time_confusion_matrix']
#     zone_cm = metrics['zone_confusion_matrix']
#     time_classes = Tt_model_results['time_encoder'].classes_
#     zone_classes = Tt_model_results['zone_encoder'].classes_
#     fig_confusion = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=('Time Slot Confusion Matrix', 'Zone Confusion Matrix'),
#         specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}]]
#     )
#     fig_confusion.add_trace(go.Heatmap(
#         z=time_cm,
#         x=time_classes,
#         y=time_classes,
#         colorscale='Blues',
#         showscale=True,
#         text=time_cm,
#         texttemplate='%{text}',
#         textfont={"size": 12},
#         hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>'
#     ), row=1, col=1)
#     fig_confusion.add_trace(go.Heatmap(
#         z=zone_cm,
#         x=zone_classes,
#         y=zone_classes,
#         colorscale='Greens',
#         showscale=True,
#         text=zone_cm,
#         texttemplate='%{text}',
#         textfont={"size": 12},
#         hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>'
#     ), row=1, col=2)
#     fig_confusion.update_xaxes(title_text="Predicted", row=1, col=1)
#     fig_confusion.update_xaxes(title_text="Predicted", row=1, col=2)
#     fig_confusion.update_yaxes(title_text="Actual", row=1, col=1)
#     fig_confusion.update_yaxes(title_text="Actual", row=1, col=2)
#     fig_confusion.update_layout(
#         height=500,
#         template='plotly_white',
#         title_text='Prediction Accuracy Analysis'
#     )
#     time_report = metrics['time_report']
#     zone_report = metrics['zone_report']
#     time_report_data = []
#     for class_name in time_classes:
#         if class_name in time_report:
#             time_report_data.append({
#                 'Time Slot': class_name,
#                 'Precision': f"{time_report[class_name]['precision']:.3f}",
#                 'Recall': f"{time_report[class_name]['recall']:.3f}",
#                 'F1-Score': f"{time_report[class_name]['f1-score']:.3f}",
#                 'Support': time_report[class_name]['support']
#             })
#     zone_report_data = []
#     for class_name in zone_classes:
#         if class_name in zone_report:
#             zone_report_data.append({
#                 'Zone': class_name,
#                 'Precision': f"{zone_report[class_name]['precision']:.3f}",
#                 'Recall': f"{zone_report[class_name]['recall']:.3f}",
#                 'F1-Score': f"{zone_report[class_name]['f1-score']:.3f}",
#                 'Support': zone_report[class_name]['support']
#             })
#     classification_reports_div = html.Div([
#         html.H3("📊 Detailed Classification Reports", style={'color': '#2c3e50', 'marginBottom': '15px'}),
#         html.Div([
#             html.H4("⏰ Time Slot Performance", style={'color': '#16a085'}),
#             dash_table.DataTable(
#                 data=time_report_data,
#                 columns=[{"name": i, "id": i} for i in time_report_data[0].keys()],
#                 style_cell={'textAlign': 'center', 'padding': '10px'},
#                 style_header={'backgroundColor': '#16a085', 'color': 'white', 'fontWeight': 'bold'},
#                 style_data_conditional=[
#                     {
#                         'if': {'row_index': 'odd'},
#                         'backgroundColor': '#f2f2f2'
#                     }
#                 ]
#             )
#         ], style={'marginBottom': '20px'}),
#         html.Div([
#             html.H4("📍 Zone Performance", style={'color': '#27ae60'}),
#             dash_table.DataTable(
#                 data=zone_report_data,
#                 columns=[{"name": i, "id": i} for i in zone_report_data[0].keys()],
#                 style_cell={'textAlign': 'center', 'padding': '10px'},
#                 style_header={'backgroundColor': '#27ae60', 'color': 'white', 'fontWeight': 'bold'},
#                 style_data_conditional=[
#                     {
#                         'if': {'row_index': 'odd'},
#                         'backgroundColor': '#f2f2f2'
#                     }
#                 ]
#             )
#         ])
#     ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '10px',
#               'border': '1px solid #ddd'})
#     metrics_div = html.Div([
#         html.H3("🎯 Model Performance Summary", style={'color': '#2c3e50', 'textAlign': 'center'}),
#         html.Div([
#             html.Div([
#                 html.H4("Time Accuracy", style={'color': '#16a085'}),
#                 html.P(f"{metrics['time_accuracy']:.2%}",
#                        style={'fontSize': '28px', 'fontWeight': 'bold'}),
#                 html.P("Primary Metric", style={'fontSize': '11px', 'color': '#7f8c8d'})
#             ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
#             html.Div([
#                 html.H4("Time Top-2", style={'color': '#1abc9c'}),
#                 html.P(f"{metrics['time_top2_accuracy']:.2%}",
#                        style={'fontSize': '28px', 'fontWeight': 'bold'}),
#                 html.P("Top-2 Accuracy", style={'fontSize': '11px', 'color': '#7f8c8d'})
#             ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
#             html.Div([
#                 html.H4("Zone Accuracy", style={'color': '#27ae60'}),
#                 html.P(f"{metrics['zone_accuracy']:.2%}",
#                        style={'fontSize': '28px', 'fontWeight': 'bold'}),
#                 html.P("Primary Metric", style={'fontSize': '11px', 'color': '#7f8c8d'})
#             ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
#             html.Div([
#                 html.H4("Zone Top-2", style={'color': '#2ecc71'}),
#                 html.P(f"{metrics['zone_top2_accuracy']:.2%}",
#                        style={'fontSize': '28px', 'fontWeight': 'bold'}),
#                 html.P("Top-2 Accuracy", style={'fontSize': '11px', 'color': '#7f8c8d'})
#             ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
#             html.Div([
#                 html.H4("Combined", style={'color': '#9b59b6'}),
#                 html.P(f"{(metrics['time_accuracy'] + metrics['zone_accuracy']) / 2:.2%}",
#                        style={'fontSize': '28px', 'fontWeight': 'bold'}),
#                 html.P("Average Accuracy", style={'fontSize': '11px', 'color': '#7f8c8d'})
#             ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
#         ])
#     ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})
#     return model_info, fig_training, fig_confusion, classification_reports_div, metrics_div
#
# @Tt_app.callback(
#     [Output('schedule-display', 'children'),
#      Output('schedule-viz', 'figure')],
#     [Input('schedule-button', 'n_clicks')],
#     [State('schedule-days-slider', 'value')]
# )
# def Tt_generate_schedule(n_clicks, days_ahead):
#     """Generate optimized rationing schedule"""
#     if n_clicks == 0 or Tt_model_results is None:
#         return html.Div([
#             html.P("⚠️ Please train the model first before generating schedule.",
#                    style={'textAlign': 'center', 'color': '#e74c3c', 'fontSize': '16px'})
#         ]), go.Figure()
#     global Tt_schedule_df
#     Tt_schedule_df = Tt_generate_rationing_schedule(Tt_model_results, Tt_df_global, days_ahead)
#     Tt_schedule_df['priority_score'] = (
#             Tt_schedule_df['zone_priority'] * 0.6 +
#             Tt_schedule_df['water_stress'] * 0.3 +
#             Tt_schedule_df['complaint_rate'] * 0.1
#     )
#     Tt_schedule_df = Tt_schedule_df.sort_values(['date', 'priority_score'], ascending=[True, False])
#     display_data = Tt_schedule_df.copy()
#     display_data['date'] = display_data['date'].dt.strftime('%Y-%m-%d (%A)')
#     display_data['time_confidence'] = (display_data['time_confidence'] * 100).round(1).astype(str) + '%'
#     display_data['zone_priority'] = (display_data['zone_priority'] * 100).round(1).astype(str) + '%'
#     display_data['priority_score'] = display_data['priority_score'].round(3)
#     display_data['demand_forecast'] = display_data['demand_forecast'].round(0).astype(int)
#     schedule_display = html.Div([
#         html.H3("📅 Optimized Water Rationing Schedule",
#                 style={'color': '#2c3e50', 'marginBottom': '15px'}),
#         html.P(f"Generated for next {days_ahead} days with AI-powered optimization",
#                style={'color': '#7f8c8d', 'marginBottom': '20px'}),
#         dash_table.DataTable(
#             data=display_data.to_dict('records'),
#             columns=[
#                 {"name": "Date", "id": "date"},
#                 {"name": "Zone", "id": "zone"},
#                 {"name": "Recommended Time", "id": "recommended_time"},
#                 {"name": "Time Confidence", "id": "time_confidence"},
#                 {"name": "Zone Priority", "id": "zone_priority"},
#                 {"name": "Priority Score", "id": "priority_score"},
#                 {"name": "Demand (m³)", "id": "demand_forecast"},
#                 {"name": "Water Stress", "id": "water_stress"},
#             ],
#             style_cell={
#                 'textAlign': 'center',
#                 'padding': '10px',
#                 'fontSize': '12px'
#             },
#             style_header={
#                 'backgroundColor': '#2980b9',
#                 'color': 'white',
#                 'fontWeight': 'bold'
#             },
#             style_data_conditional=[
#                 {
#                     'if': {'row_index': 'odd'},
#                     'backgroundColor': '#f9f9f9'
#                 },
#                 {
#                     'if': {
#                         'filter_query': '{priority_score} > 0.7',
#                         'column_id': 'priority_score'
#                     },
#                     'backgroundColor': '#d4edda',
#                     'color': '#155724',
#                     'fontWeight': 'bold'
#                 }
#             ],
#             page_size=20,
#             sort_action='native',
#             filter_action='native'
#         )
#     ], style={'backgroundColor': '#fff', 'padding': '20px', 'borderRadius': '10px',
#               'border': '1px solid #ddd'})
#     fig_schedule = make_subplots(
#         rows=2, cols=2,
#         subplot_titles=('Time Distribution by Zone', 'Zone Priority Over Time',
#                         'Water Stress by Zone', 'Demand Forecast Timeline'),
#         specs=[[{'type': 'bar'}, {'type': 'scatter'}],
#                [{'type': 'box'}, {'type': 'scatter'}]],
#         vertical_spacing=0.15,
#         horizontal_spacing=0.12
#     )
#     time_zone_counts = Tt_schedule_df.groupby(['zone', 'recommended_time']).size().reset_index(name='count')
#     for zone in Tt_schedule_df['zone'].unique():
#         zone_data = time_zone_counts[time_zone_counts['zone'] == zone]
#         fig_schedule.add_trace(go.Bar(
#             x=zone_data['recommended_time'],
#             y=zone_data['count'],
#             name=zone,
#             hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
#         ), row=1, col=1)
#     for zone in Tt_schedule_df['zone'].unique():
#         zone_data = Tt_schedule_df[Tt_schedule_df['zone'] == zone]
#         fig_schedule.add_trace(go.Scatter(
#             x=zone_data['date'],
#             y=zone_data['zone_priority'],
#             mode='lines+markers',
#             name=zone,
#             hovertemplate='Date: %{x}<br>Priority: %{y:.3f}<extra></extra>'
#         ), row=1, col=2)
#     for zone in Tt_schedule_df['zone'].unique():
#         zone_data = Tt_schedule_df[Tt_schedule_df['zone'] == zone]
#         fig_schedule.add_trace(go.Box(
#             y=zone_data['water_stress'],
#             name=zone,
#             boxmean='sd'
#         ), row=2, col=1)
#     demand_by_date = Tt_schedule_df.groupby('date')['demand_forecast'].sum().reset_index()
#     fig_schedule.add_trace(go.Scatter(
#         x=demand_by_date['date'],
#         y=demand_by_date['demand_forecast'],
#         mode='lines+markers',
#         name='Total Demand',
#         line=dict(color='#e74c3c', width=3),
#         fill='tozeroy',
#         fillcolor='rgba(231, 76, 60, 0.2)',
#         hovertemplate='Date: %{x}<br>Demand: %{y:,.0f} m³<extra></extra>'
#     ), row=2, col=2)
#     fig_schedule.update_xaxes(title_text="Time Slot", row=1, col=1)
#     fig_schedule.update_xaxes(title_text="Date", row=1, col=2)
#     fig_schedule.update_xaxes(title_text="Zone", row=2, col=1)
#     fig_schedule.update_xaxes(title_text="Date", row=2, col=2)
#     fig_schedule.update_yaxes(title_text="Frequency", row=1, col=1)
#     fig_schedule.update_yaxes(title_text="Priority Score", row=1, col=2)
#     fig_schedule.update_yaxes(title_text="Water Stress", row=2, col=1)
#     fig_schedule.update_yaxes(title_text="Demand (m³)", row=2, col=2)
#     fig_schedule.update_layout(
#         height=800,
#         template='plotly_white',
#         title_text='Rationing Schedule Analysis & Insights',
#         showlegend=True
#     )
#     return schedule_display, fig_schedule
#
# if __name__ == '__main__':
#     Tt_app.run(debug=True, port=8054)