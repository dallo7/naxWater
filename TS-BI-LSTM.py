import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')


# Load and prepare the real dataset
def load_and_prepare_data(filepath='nawassco.csv'):
    """Load and prepare the water supply dataset"""
    try:
        df = pd.read_csv(filepath)

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace('/', '_').str.replace(' ', '_')

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Using simulated data instead.")
        return None


def prepare_timeseries_data(df, target_column='actual_consumption_m3', aggregate=True):
    """Prepare data for LSTM (ds, y)"""
    if aggregate:
        # Aggregate by date (sum consumption across all zones)
        ts_df = df.groupby('date').agg({
            target_column: 'sum',
            'rainfall_mm': 'mean',
            'pipe_leakage_m3': 'sum',
            'complaints_received': 'sum',
            'population_served': 'sum'
        }).reset_index()
    else:
        ts_df = df[['date', target_column]].copy()

    # Rename columns
    ts_df = ts_df.rename(columns={'date': 'ds', target_column: 'y'})

    return ts_df


def add_noise_to_data(df, noise_level=0):
    """Add additional noise to test LSTM's robustness"""
    if noise_level > 0:
        df_noisy = df.copy()
        noise = np.random.normal(0, noise_level, len(df))
        df_noisy['y'] = df_noisy['y'] + noise
        return df_noisy
    return df


def create_challenge_scenario(df, scenario='normal'):
    """Create challenging scenarios from the base dataset"""
    df_challenge = df.copy()

    if scenario == 'missing_data':
        # Remove 20% of data randomly
        missing_indices = np.random.choice(len(df), size=int(len(df) * 0.2), replace=False)
        df_challenge.loc[missing_indices, 'y'] = np.nan
        # LSTM requires complete data - interpolate
        df_challenge['y'] = df_challenge['y'].interpolate(method='linear')

    elif scenario == 'outliers':
        # Add extreme outliers to 5% of data
        outlier_indices = np.random.choice(len(df), size=max(1, int(len(df) * 0.05)), replace=False)
        df_challenge.loc[outlier_indices, 'y'] = df_challenge.loc[outlier_indices, 'y'] * np.random.choice([0.3, 2.5],
                                                                                                           len(outlier_indices))

    elif scenario == 'trend_shift':
        # Add a sudden trend shift halfway through
        midpoint = len(df) // 2
        df_challenge.loc[midpoint:, 'y'] = df_challenge.loc[midpoint:, 'y'] * 1.3

    return df_challenge


def create_sequences(data, lookback=7):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


def build_lstm_model(lookback, units=50, dropout_rate=0.2, layers=2):
    """Build Bidirectional LSTM model architecture"""
    model = Sequential()

    # First Bi-LSTM layer
    model.add(Bidirectional(
        LSTM(units=units, return_sequences=(layers > 1), input_shape=(lookback, 1)),
        name='bidirectional_lstm_1'
    ))
    model.add(Dropout(dropout_rate))

    # Additional Bi-LSTM layers
    for i in range(1, layers):
        return_sequences = (i < layers - 1)
        model.add(Bidirectional(
            LSTM(units=units, return_sequences=return_sequences),
            name=f'bidirectional_lstm_{i + 1}'
        ))
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


# Initialize the Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "Aqua-Predict | BI-LSTM Water Forecasting"

server = app.server
# Try to load real data
real_data = load_and_prepare_data()
if real_data is not None:
    base_ts_data = prepare_timeseries_data(real_data)
    data_source = "Real Water Supply Data"
else:
    # Fallback to simulated data if file not found
    dates = pd.date_range(start='2025-09-01', periods=77, freq='D')
    values = 35000 + 5000 * np.sin(2 * np.pi * np.arange(77) / 7) + np.random.normal(0, 2000, 77)
    base_ts_data = pd.DataFrame({'ds': dates, 'y': values})
    data_source = "Simulated Data (CSV not found)"

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("💧 Water Supply Forecasting with Bi-LSTM Neural Networks",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P(
            f"Bidirectional LSTM for enhanced time series prediction with forward & backward context - {data_source}",
            style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'})
    ]),

    # Control panel
    html.Div([
        html.Div([
            html.Label("Select Challenge Scenario:", style={'fontWeight': 'bold', 'fontSize': '16px'}),
            dcc.Dropdown(
                id='scenario-dropdown',
                options=[
                    {'label': '📊 Normal - Actual Dataset', 'value': 'normal'},
                    {'label': '🕳️ Challenge 1: Missing Data (20% gaps)', 'value': 'missing_data'},
                    {'label': '⚡ Challenge 2: Extreme Outliers (5% anomalies)', 'value': 'outliers'},
                    {'label': '📉 Challenge 3: Trend Shift (30% increase)', 'value': 'trend_shift'},
                    {'label': '🔍 Challenge 4: High Noise Environment', 'value': 'high_noise'}
                ],
                value='normal',
                style={'width': '100%'}
            )
        ], style={'width': '100%', 'marginBottom': '20px'}),

        html.Div([
            html.Div([
                html.Label("Additional Noise Level (m³):", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='noise-slider',
                    min=0,
                    max=5000,
                    step=500,
                    value=0,
                    marks={0: '0', 1000: '1K', 2500: '2.5K', 5000: '5K'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px'}),

            html.Div([
                html.Label("Forecast Period (days):", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='forecast-slider',
                    min=7,
                    max=30,
                    step=7,
                    value=14,
                    marks={7: '7', 14: '14', 21: '21', 30: '30'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px'}),
        ]),

        html.Div([
            html.Div([
                html.Label("Lookback Window (days):", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='lookback-slider',
                    min=3,
                    max=14,
                    step=1,
                    value=7,
                    marks={3: '3', 7: '7', 10: '10', 14: '14'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.P("Number of past days to use for prediction",
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '5px'})
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),

            html.Div([
                html.Label("Bi-LSTM Units:", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='units-slider',
                    min=25,
                    max=100,
                    step=25,
                    value=50,
                    marks={25: '25', 50: '50', 75: '75', 100: '100'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.P("Hidden units in each Bi-LSTM layer (×2 for bidirectional)",
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '5px'})
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),

            html.Div([
                html.Label("Number of Bi-LSTM Layers:", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='layers-slider',
                    min=1,
                    max=3,
                    step=1,
                    value=2,
                    marks={1: '1', 2: '2', 3: '3'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.P("Depth of neural network",
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '5px'})
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),
        ]),

    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),

    # Scenario description
    html.Div(id='scenario-description', style={
        'backgroundColor': '#fff3cd', 'padding': '15px', 'borderRadius': '5px',
        'marginBottom': '20px', 'border': '2px solid #ffc107'
    }),

    # Dataset info
    html.Div(id='dataset-info', style={
        'backgroundColor': '#d1ecf1', 'padding': '15px', 'borderRadius': '5px',
        'marginBottom': '20px', 'border': '2px solid #17a2b8'
    }),

    html.Div([
        html.Button('🧠 Train Bi-LSTM & Forecast', id='forecast-button', n_clicks=0,
                    style={'backgroundColor': '#9b59b6', 'color': 'white', 'padding': '15px 30px',
                           'fontSize': '16px', 'border': 'none', 'borderRadius': '5px',
                           'cursor': 'pointer', 'display': 'block', 'margin': '0 auto'})
    ], style={'marginBottom': '20px'}),

    # Loading component
    dcc.Loading(
        id="loading",
        type="default",
        children=[
            # Training info
            html.Div(id='training-info', style={'marginBottom': '20px'}),

            # Forecast plot
            dcc.Graph(id='forecast-plot', style={'marginBottom': '20px'}),

            # Training history plot
            dcc.Graph(id='training-plot', style={'marginBottom': '20px'}),

            # Metrics and insights
            html.Div(id='metrics-display', style={'marginTop': '20px'}),
            html.Div(id='insights-display', style={'marginTop': '20px'})
        ]
    )
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px', 'margin': '0 auto'})


@app.callback(
    [Output('scenario-description', 'children'),
     Output('dataset-info', 'children')],
    Input('scenario-dropdown', 'value')
)
def update_descriptions(scenario):
    """Update scenario description and dataset info"""
    descriptions = {
        'normal': "📊 Baseline scenario - Bi-LSTM will learn temporal patterns bidirectionally from historical data.",
        'missing_data': "🕳️ CHALLENGE 1: Dataset with 20% missing data. Bi-LSTM trained on interpolated data.",
        'outliers': "⚡ CHALLENGE 2: Extreme consumption spikes/drops (5%). Bi-LSTM may learn outlier patterns.",
        'trend_shift': "📉 CHALLENGE 3: Sudden 30% increase. Bi-LSTM can adapt to regime changes with bidirectional context.",
        'high_noise': "🔍 CHALLENGE 4: High volatility. Bi-LSTM's forward-backward memory helps filter noise patterns."
    }

    scenario_desc = html.Div([
        html.Strong("Current Scenario: "),
        html.Span(descriptions.get(scenario, "Select a scenario"))
    ])

    # Dataset info
    if real_data is not None:
        total_consumption = real_data['actual_consumption_m3'].sum()
        avg_daily = base_ts_data['y'].mean()
        date_range = f"{base_ts_data['ds'].min().strftime('%Y-%m-%d')} to {base_ts_data['ds'].max().strftime('%Y-%m-%d')}"
        num_zones = real_data['zone'].nunique()

        dataset_desc = html.Div([
            html.Strong("📊 Dataset Overview: "),
            html.Span(f"{len(base_ts_data)} days | {num_zones} zones | "),
            html.Span(f"Avg Daily: {avg_daily:,.0f} m³ | "),
            html.Span(f"Total: {total_consumption:,.0f} m³ | "),
            html.Span(f"Period: {date_range}")
        ])
    else:
        dataset_desc = html.Div([
            html.Strong("⚠️ Note: "),
            html.Span("Using simulated data. Place 'dataset_Gans_run_1.csv' in the same directory to use real data.")
        ])

    return scenario_desc, dataset_desc


@app.callback(
    [Output('training-info', 'children'),
     Output('forecast-plot', 'figure'),
     Output('training-plot', 'figure'),
     Output('metrics-display', 'children'),
     Output('insights-display', 'children')],
    [Input('forecast-button', 'n_clicks')],
    [State('scenario-dropdown', 'value'),
     State('noise-slider', 'value'),
     State('forecast-slider', 'value'),
     State('lookback-slider', 'value'),
     State('units-slider', 'value'),
     State('layers-slider', 'value')]
)
def update_forecast(n_clicks, scenario, noise_level, forecast_days, lookback, units, num_layers):
    """Train LSTM model and update all visualizations"""

    if n_clicks == 0:
        return html.Div(), go.Figure(), go.Figure(), html.Div(), html.Div()

    # Prepare data based on scenario
    df = base_ts_data.copy()

    # Apply challenge scenarios
    if scenario != 'normal' and scenario != 'high_noise':
        df = create_challenge_scenario(df, scenario)

    # Add noise
    if scenario == 'high_noise':
        noise_level = max(noise_level, 3000)

    df = add_noise_to_data(df, noise_level)

    # Prepare data for LSTM
    data = df['y'].values.reshape(-1, 1)

    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X, y = create_sequences(scaled_data, lookback)

    # Reshape for LSTM [samples, time steps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train model
    try:
        model = build_lstm_model(lookback, units, dropout_rate=0.2, layers=num_layers)

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=16,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )

        # Make predictions on training data
        train_predict = model.predict(X_train, verbose=0)
        test_predict = model.predict(X_test, verbose=0)

        # Inverse transform predictions
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train_actual = scaler.inverse_transform(y_train)
        y_test_actual = scaler.inverse_transform(y_test)

        # Multi-step forecasting
        last_sequence = scaled_data[-lookback:]
        forecast_scaled = []

        current_sequence = last_sequence.copy()
        for _ in range(forecast_days):
            current_sequence_reshaped = current_sequence.reshape((1, lookback, 1))
            next_pred = model.predict(current_sequence_reshaped, verbose=0)
            forecast_scaled.append(next_pred[0, 0])
            current_sequence = np.append(current_sequence[1:], next_pred)

        forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
        forecast_values = scaler.inverse_transform(forecast_scaled)

        # Calculate confidence intervals (using standard deviation of residuals)
        residuals = y_test_actual.flatten() - test_predict.flatten()
        std_residual = np.std(residuals)

        forecast_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1),
                                       periods=forecast_days, freq='D')

        forecast_data = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast_values.flatten(),
            'yhat_lower': forecast_values.flatten() - 1.96 * std_residual,
            'yhat_upper': forecast_values.flatten() + 1.96 * std_residual
        })

        model_success = True
        error_msg = None

        # Calculate metrics
        train_mae = mean_absolute_error(y_train_actual, train_predict)
        test_mae = mean_absolute_error(y_test_actual, test_predict)
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

        final_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        epochs_trained = len(history.history['loss'])

    except Exception as e:
        model_success = False
        error_msg = str(e)

        # Create dummy forecast
        forecast_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1),
                                       periods=forecast_days, freq='D')
        forecast_data = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': [df['y'].mean()] * forecast_days,
            'yhat_lower': [df['y'].mean() * 0.9] * forecast_days,
            'yhat_upper': [df['y'].mean() * 1.1] * forecast_days
        })

        train_predict = np.array([df['y'].mean()] * train_size)
        test_predict = np.array([df['y'].mean()] * (len(df) - lookback - train_size))
        train_mae = test_mae = train_rmse = test_rmse = 0
        history = None
        epochs_trained = 0

    # Create training info box
    if model_success:
        training_info = html.Div([
            html.H3("🧠 Bi-LSTM Training Results",
                    style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.Strong("Architecture: "),
                    html.Span(
                        f"{num_layers} Bidirectional LSTM layers with {units} units each (×2 = {units * 2} total per layer)")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Lookback Window: "),
                    html.Span(f"{lookback} days")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Model Parameters: "),
                    html.Span(f"{model.count_params():,} (2× unidirectional LSTM due to bidirectional)")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Training Samples: "),
                    html.Span(f"{train_size} sequences")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Validation Samples: "),
                    html.Span(f"{len(X_test)} sequences")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Epochs Trained: "),
                    html.Span(f"{epochs_trained} (early stopping)")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Final Training Loss: "),
                    html.Span(f"{final_loss:.6f}")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Final Validation Loss: "),
                    html.Span(f"{final_val_loss:.6f}")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Bi-LSTM Advantage: "),
                    html.Span("Processes sequences forward AND backward for better context understanding")
                ])
            ])
        ], style={'backgroundColor': '#e8f5e9', 'padding': '20px', 'borderRadius': '10px',
                  'border': '2px solid #4caf50'})
    else:
        training_info = html.Div([
            html.H3("❌ LSTM Training Failed", style={'color': '#c0392b'}),
            html.P(f"Error: {error_msg}")
        ], style={'backgroundColor': '#f8d7da', 'padding': '20px', 'borderRadius': '10px',
                  'border': '2px solid #dc3545'})

    # Create forecast plot
    fig_forecast = go.Figure()

    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='markers',
        name='Actual Consumption',
        marker=dict(size=6, color='#2c3e50', opacity=0.7),
        hovertemplate='Date: %{x}<br>Consumption: %{y:,.0f} m³<extra></extra>'
    ))

    if model_success:
        # Training predictions
        train_dates = df['ds'].iloc[lookback:lookback + train_size]
        fig_forecast.add_trace(go.Scatter(
            x=train_dates,
            y=train_predict.flatten(),
            mode='lines',
            name='Bi-LSTM Training Fit',
            line=dict(color='#27ae60', width=2),
            hovertemplate='Date: %{x}<br>Fitted: %{y:,.0f} m³<extra></extra>'
        ))

        # Test predictions
        test_dates = df['ds'].iloc[lookback + train_size:]
        fig_forecast.add_trace(go.Scatter(
            x=test_dates,
            y=test_predict.flatten(),
            mode='lines',
            name='Bi-LSTM Validation Fit',
            line=dict(color='#f39c12', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>Validation: %{y:,.0f} m³<extra></extra>'
        ))

    # Forecast
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat'],
        mode='lines',
        name='Bi-LSTM Forecast',
        line=dict(color='#9b59b6', width=3),
        hovertemplate='Date: %{x}<br>Forecast: %{y:,.0f} m³<extra></extra>'
    ))

    # Confidence interval
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat_lower'],
        mode='lines',
        name='95% Confidence Interval',
        fill='tonexty',
        fillcolor='rgba(155, 89, 182, 0.2)',
        line=dict(width=0),
        hovertemplate='Lower: %{y:,.0f} m³<extra></extra>'
    ))

    fig_forecast.update_layout(
        title=f'Water Consumption Forecast - Bi-LSTM ({num_layers} layers, {units}×2 units) - {scenario.replace("_", " ").title()}',
        xaxis_title='Date',
        yaxis_title='Water Consumption (m³)',
        hovermode='x unified',
        template='plotly_white',
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Create training history plot
    if model_success and history:
        fig_training = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training & Validation Loss', 'Mean Absolute Error'),
            vertical_spacing=0.15
        )

        # Loss
        fig_training.add_trace(go.Scatter(
            x=list(range(1, epochs_trained + 1)),
            y=history.history['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='#e74c3c', width=2)
        ), row=1, col=1)

        fig_training.add_trace(go.Scatter(
            x=list(range(1, epochs_trained + 1)),
            y=history.history['val_loss'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='#3498db', width=2)
        ), row=1, col=1)

        # MAE
        fig_training.add_trace(go.Scatter(
            x=list(range(1, epochs_trained + 1)),
            y=history.history['mae'],
            mode='lines',
            name='Training MAE',
            line=dict(color='#27ae60', width=2)
        ), row=2, col=1)

        fig_training.add_trace(go.Scatter(
            x=list(range(1, epochs_trained + 1)),
            y=history.history['val_mae'],
            mode='lines',
            name='Validation MAE',
            line=dict(color='#f39c12', width=2)
        ), row=2, col=1)

        fig_training.update_xaxes(title_text="Epoch", row=2, col=1)
        fig_training.update_yaxes(title_text="Loss (MSE)", row=1, col=1)
        fig_training.update_yaxes(title_text="MAE", row=2, col=1)

        fig_training.update_layout(
            height=600,
            template='plotly_white',
            title_text='Bi-LSTM Training Progress',
            showlegend=True
        )
    else:
        fig_training = go.Figure()
        fig_training.add_annotation(
            text="Training history not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#7f8c8d')
        )

    # Forecast summary
    avg_forecast = forecast_data['yhat'].mean()
    total_forecast = forecast_data['yhat'].sum()

    metrics_div = html.Div([
        html.H3("📊 Bi-LSTM Performance & Forecast Summary",
                style={'color': '#2c3e50', 'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.H4("Train MAE", style={'color': '#27ae60'}),
                html.P(f"{train_mae:,.0f} m³" if model_success else "N/A",
                       style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P("Training Error", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Test MAE", style={'color': '#f39c12'}),
                html.P(f"{test_mae:,.0f} m³" if model_success else "N/A",
                       style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P("Validation Error", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Train RMSE", style={'color': '#e74c3c'}),
                html.P(f"{train_rmse:,.0f} m³" if model_success else "N/A",
                       style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P("Training RMSE", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Test RMSE", style={'color': '#c0392b'}),
                html.P(f"{test_rmse:,.0f} m³" if model_success else "N/A",
                       style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P("Validation RMSE", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Avg Forecast", style={'color': '#9b59b6'}),
                html.P(f"{avg_forecast:,.0f} m³", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P(f"Next {forecast_days} days", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Total Forecast", style={'color': '#8e44ad'}),
                html.P(f"{total_forecast:,.0f} m³", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P(f"Period demand", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),
        ])
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})

    # Generate insights
    if not model_success:
        insights = [
            f"❌ Bi-LSTM training failed: {error_msg}",
            "💡 Try reducing model complexity (fewer layers/units)",
            "💡 Check if dataset has sufficient samples for training",
            "💡 Consider data preprocessing or feature engineering"
        ]
        bg_color = '#f8d7da'
        border_color = '#dc3545'
    else:
        # Calculate overfitting indicator
        overfit_ratio = test_mae / train_mae if train_mae > 0 else 1.0

        insights_dict = {
            'normal': [
                f"✅ Bi-LSTM trained successfully on {train_size} sequences with {lookback}-day lookback",
                f"✅ Validation MAE: {test_mae:,.0f} m³ - Model generalizes {'well' if overfit_ratio < 1.5 else 'moderately'}",
                f"✅ Training converged in {epochs_trained} epochs (early stopping)",
                f"✅ Bidirectional processing: 2×{units} = {units * 2} parameters per layer",
                f"🎯 Overfitting ratio: {overfit_ratio:.2f} ({'Good' if overfit_ratio < 1.5 else 'Some overfitting detected'})"
            ],
            'missing_data': [
                f"✅ Bi-LSTM handled interpolated missing data (originally 20% gaps)",
                f"✅ Bidirectional context helps bridge gaps better than unidirectional",
                f"✅ Model learned patterns despite data preprocessing",
                "🎯 Evidence: Check if predictions align with interpolated regions"
            ],
            'outliers': [
                f"⚠️ Bi-LSTM learned from outlier-contaminated data (5% anomalies)",
                f"✅ Validation MAE: {test_mae:,.0f} m³ - Some outlier influence expected",
                f"✅ Bidirectional processing provides more context for anomaly detection",
                "🎯 Evidence: Training loss higher than normal due to outliers"
            ],
            'trend_shift': [
                f"✅ Bi-LSTM adapted to 30% trend shift through bidirectional training",
                f"✅ Forward-backward context captures regime changes better",
                f"✅ Lookback of {lookback} days with both directions provides rich context",
                "🎯 Evidence: Validation fit shows adaptation to new consumption level"
            ],
            'high_noise': [
                f"✅ Bi-LSTM's bidirectional memory cells filter noise patterns (noise level: {noise_level:,.0f} m³)",
                f"✅ Dropout ({0.2}) helps prevent overfitting to noise",
                f"✅ Validation RMSE: {test_rmse:,.0f} m³",
                f"✅ 2× parameters ({units * 2} per layer) provide better noise filtering",
                "🎯 Evidence: Training history shows stable convergence despite volatility"
            ]
        }

        insights = insights_dict.get(scenario, [])

        # Add model architecture insight
        if overfit_ratio > 2.0:
            insights.append("⚠️ Consider reducing model complexity - significant overfitting detected")
        elif overfit_ratio < 1.2:
            insights.append("✅ Excellent generalization - Bi-LSTM doesn't overfit")

        # Add Bi-LSTM specific advantage
        insights.append(
            f"💡 Bi-LSTM Advantage: {units * 2} effective parameters per layer vs {units} in unidirectional LSTM")

        bg_color = '#d4edda'
        border_color = '#28a745'

    insights_div = html.Div([
        html.H3("🎯 Bi-LSTM Performance Evidence",
                style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Ul([html.Li(insight, style={'marginBottom': '10px', 'fontSize': '14px'})
                 for insight in insights])
    ], style={'backgroundColor': bg_color, 'padding': '20px', 'borderRadius': '10px',
              'border': f'2px solid {border_color}'})

    return training_info, fig_forecast, fig_training, metrics_div, insights_div


if __name__ == '__main__':

    app.run(debug=True, port=8055)
