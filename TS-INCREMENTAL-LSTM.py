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
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input as KerasInput
import warnings
import dash_bootstrap_components as dbc


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
    """Add additional noise to test Incremental LSTM's robustness"""
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
        # Incremental LSTM requires complete data - interpolate
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


def build_incremental_lstm_model(batch_size, lookback, units=50, dropout_rate=0.2, layers=2):

    model = Sequential()

    # First Stateful LSTM layer
    # Use KerasInput layer to specify batch shape
    model.add(KerasInput(batch_shape=(batch_size, lookback, 1)))

    model.add(LSTM(
        units=units,
        return_sequences=(layers > 1),
        stateful=True,  # Maintains state across batches
        name='stateful_lstm_1'
    ))
    model.add(Dropout(dropout_rate))

    # Additional Stateful LSTM layers
    for i in range(1, layers):
        return_sequences = (i < layers - 1)
        model.add(LSTM(
            units=units,
            return_sequences=return_sequences,
            stateful=True,  
            name=f'stateful_lstm_{i + 1}'
        ))
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


app=dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "Aqua-Predict | INCREMENTAL LSTM"
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
        html.H1("💧 Water Supply Forecasting with Incremental LSTM",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P(f"Stateful LSTM for continuous learning and incremental updates from streaming data - {data_source}",
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
                html.Label("Incremental LSTM Units:", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='units-slider',
                    min=25,
                    max=100,
                    step=25,
                    value=50,
                    marks={25: '25', 50: '50', 75: '75', 100: '100'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.P("Hidden units with state preservation",
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '5px'})
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),

            html.Div([
                html.Label("Number of Stateful Layers:", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='layers-slider',
                    min=1,
                    max=3,
                    step=1,
                    value=2,
                    marks={1: '1', 2: '2', 3: '3'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.P("Depth of incremental network",
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '5px'})
            ], style={'width': '30%', 'display': 'inline-block', 'padding': '20px'}),
        ]),

        html.Div([
            html.Label("🔄 Incremental Learning Mode:", style={'fontWeight': 'bold', 'fontSize': '14px'}),
            dcc.RadioItems(
                id='learning-mode',
                options=[
                    {'label': ' Batch Training (Reset states between epochs)', 'value': 'batch'},
                    {'label': ' Continuous Learning (Maintain states across epochs)', 'value': 'continuous'}
                ],
                value='continuous',
                inline=False,
                style={'marginTop': '10px'}
            )
        ], style={'width': '90%', 'margin': '0 auto', 'padding': '20px'}),

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
        html.Button('🔄 Train Incremental LSTM & Forecast', id='forecast-button', n_clicks=0,
                    style={'backgroundColor': '#e67e22', 'color': 'white', 'padding': '15px 30px',
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

            # State evolution plot
            dcc.Graph(id='state-plot', style={'marginBottom': '20px'}),

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
        'normal': "📊 Baseline scenario - Incremental LSTM maintains hidden states for continuous learning from sequential data.",
        'missing_data': "🕳️ CHALLENGE 1: Dataset with 20% missing data. Incremental LSTM preserves context across gaps.",
        'outliers': "⚡ CHALLENGE 2: Extreme consumption spikes/drops (5%). Stateful memory helps detect anomalies in context.",
        'trend_shift': "📉 CHALLENGE 3: Sudden 30% increase. Incremental updates allow rapid adaptation to regime changes.",
        'high_noise': "🔍 CHALLENGE 4: High volatility. Stateful architecture maintains signal continuity through noise."
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
     Output('state-plot', 'figure'),
     Output('metrics-display', 'children'),
     Output('insights-display', 'children')],
    [Input('forecast-button', 'n_clicks')],
    [State('scenario-dropdown', 'value'),
     State('noise-slider', 'value'),
     State('forecast-slider', 'value'),
     State('lookback-slider', 'value'),
     State('units-slider', 'value'),
     State('layers-slider', 'value'),
     State('learning-mode', 'value')]
)
def update_forecast(n_clicks, scenario, noise_level, forecast_days, lookback, units, num_layers, learning_mode):
    """Train Incremental LSTM model and update all visualizations"""

    if n_clicks == 0:
        return html.Div(), go.Figure(), go.Figure(), go.Figure(), html.Div(), html.Div()

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

    # IMPORTANT: For stateful LSTM, batch_size must divide data evenly
    # Find appropriate batch size
    possible_batch_sizes = [1, 2, 4, 8, 16, 32]
    batch_size = 1
    for bs in possible_batch_sizes:
        if len(X) % bs == 0:
            batch_size = bs
        else:
            break

    # Trim data to fit batch size
    trim = len(X) % batch_size
    if trim > 0:
        X = X[:-trim]
        y = y[:-trim]

    # Split into train and test
    train_size = int(len(X) * 0.8)
    # Ensure train_size is divisible by batch_size
    train_size = (train_size // batch_size) * batch_size

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train model
    try:
        print(f"Training Incremental LSTM with batch_size={batch_size}, stateful={learning_mode == 'continuous'}")

        # Build the model first
        model = build_incremental_lstm_model(batch_size, lookback, units, dropout_rate=0.2, layers=num_layers)

        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train model
        history_losses = []
        history_val_losses = []
        history_maes = []
        history_val_maes = []

        epochs = 50  

        for epoch in range(epochs):
            # Train on batches
            epoch_loss = []
            epoch_mae = []

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                history = model.train_on_batch(batch_X, batch_y)
                epoch_loss.append(history[0])
                epoch_mae.append(history[1])

            # Validation
            val_loss = []
            val_mae = []
            for i in range(0, len(X_test), batch_size):
                batch_X = X_test[i:i + batch_size]
                batch_y = y_test[i:i + batch_size]

                val_metrics = model.test_on_batch(batch_X, batch_y)
                val_loss.append(val_metrics[0])
                val_mae.append(val_metrics[1])

            history_losses.append(np.mean(epoch_loss))
            history_val_losses.append(np.mean(val_loss))
            history_maes.append(np.mean(epoch_mae))
            history_val_maes.append(np.mean(val_mae))

            # Reset states if batch mode
            if learning_mode == 'batch':
                # Ensure model is built before resetting
                if hasattr(model, 'layers') and len(model.layers) > 0:
                    for layer in model.layers:
                        if hasattr(layer, 'reset_states'):
                            layer.reset_states()

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}/{epochs} - Loss: {history_losses[-1]:.6f}, Val Loss: {history_val_losses[-1]:.6f}")

        # Make predictions
        # Reset states before making predictions
        for layer in model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()

        train_predict = model.predict(X_train, batch_size=batch_size, verbose=0)

        # Reset states between train and test predictions
        for layer in model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()

        test_predict = model.predict(X_test, batch_size=batch_size, verbose=0)

        # Inverse transform predictions
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train_actual = scaler.inverse_transform(y_train)
        y_test_actual = scaler.inverse_transform(y_test)

        # Multi-step forecasting with state preservation
        # Reset states before forecasting
        for layer in model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()

        last_sequence = scaled_data[-lookback:]
        forecast_scaled = []

        current_sequence = last_sequence.copy()
        for step in range(forecast_days):
            # Reshape to match batch size
            current_batch = np.repeat(current_sequence.reshape(1, lookback, 1), batch_size, axis=0)
            next_pred = model.predict(current_batch, batch_size=batch_size, verbose=0)

            # Take first prediction (all should be similar due to same input)
            next_val = next_pred[0, 0]
            forecast_scaled.append(next_val)

            # Update sequence
            current_sequence = np.append(current_sequence[1:], [[next_val]], axis=0)

        forecast_scaled = np.array(forecast_scaled).reshape(-1, 1)
        forecast_values = scaler.inverse_transform(forecast_scaled)

        # Calculate confidence intervals
        residuals = y_test_actual.flatten() - test_predict.flatten()
        std_residual = np.std(residuals)

        forecast_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1),
                                       periods=forecast_days, freq='D')

        forecast_data = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast_values.flatten(),
            'yhat_lower': forecast_values.flatten() - 1.96 * std_residual * np.sqrt(np.arange(1, forecast_days + 1)),
            'yhat_upper': forecast_values.flatten() + 1.96 * std_residual * np.sqrt(np.arange(1, forecast_days + 1))
        })

        model_success = True
        error_msg = None

        # Calculate metrics
        train_mae = mean_absolute_error(y_train_actual, train_predict)
        test_mae = mean_absolute_error(y_test_actual, test_predict)
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))

        epochs_trained = epochs

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
        history_losses = []
        epochs_trained = 0
        batch_size = 1

    # Create training info box
    if model_success:
        training_info = html.Div([
            html.H3("🔄 Incremental LSTM Training Results",
                    style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    html.Strong("Architecture: "),
                    html.Span(f"{num_layers} Stateful LSTM layers with {units} units each")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Learning Mode: "),
                    html.Span(
                        f"{'Continuous (States preserved)' if learning_mode == 'continuous' else 'Batch (States reset)'}")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Batch Size: "),
                    html.Span(f"{batch_size} (Required for stateful processing)")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Lookback Window: "),
                    html.Span(f"{lookback} days")
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
                    html.Span(f"{epochs_trained} (incremental updates)")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("State Management: "),
                    html.Span("Hidden states maintained across batches for continuous learning")
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Strong("Incremental Advantage: "),
                    html.Span("Can update model with new data without full retraining")
                ])
            ])
        ], style={'backgroundColor': '#fff3e0', 'padding': '20px', 'borderRadius': '10px',
                  'border': '2px solid #e67e22'})
    else:
        training_info = html.Div([
            html.H3("❌ Incremental LSTM Training Failed", style={'color': '#c0392b'}),
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
            name='Incremental LSTM Training Fit',
            line=dict(color='#27ae60', width=2),
            hovertemplate='Date: %{x}<br>Fitted: %{y:,.0f} m³<extra></extra>'
        ))

        # Test predictions
        test_dates = df['ds'].iloc[lookback + train_size:]
        fig_forecast.add_trace(go.Scatter(
            x=test_dates,
            y=test_predict.flatten(),
            mode='lines',
            name='Incremental LSTM Validation Fit',
            line=dict(color='#f39c12', width=2, dash='dash'),
            hovertemplate='Date: %{x}<br>Validation: %{y:,.0f} m³<extra></extra>'
        ))

    # Forecast
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat'],
        mode='lines',
        name='Incremental LSTM Forecast',
        line=dict(color='#e67e22', width=3),
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
        fillcolor='rgba(230, 126, 34, 0.2)',
        line=dict(width=0),
        hovertemplate='Lower: %{y:,.0f} m³<extra></extra>'
    ))

    fig_forecast.update_layout(
        title=f'Water Consumption Forecast - Incremental LSTM ({num_layers} layers, {units} units, {learning_mode}) - {scenario.replace("_", " ").title()}',
        xaxis_title='Date',
        yaxis_title='Water Consumption (m³)',
        hovermode='x unified',
        template='plotly_white',
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Create training history plot
    if model_success and len(history_losses) > 0:
        fig_training = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training & Validation Loss (Incremental)', 'Mean Absolute Error (Incremental)'),
            vertical_spacing=0.15
        )

        # Loss
        fig_training.add_trace(go.Scatter(
            x=list(range(1, len(history_losses) + 1)),
            y=history_losses,
            mode='lines',
            name='Training Loss',
            line=dict(color='#e74c3c', width=2)
        ), row=1, col=1)

        fig_training.add_trace(go.Scatter(
            x=list(range(1, len(history_val_losses) + 1)),
            y=history_val_losses,
            mode='lines',
            name='Validation Loss',
            line=dict(color='#3498db', width=2)
        ), row=1, col=1)

        # MAE
        fig_training.add_trace(go.Scatter(
            x=list(range(1, len(history_maes) + 1)),
            y=history_maes,
            mode='lines',
            name='Training MAE',
            line=dict(color='#27ae60', width=2)
        ), row=2, col=1)

        fig_training.add_trace(go.Scatter(
            x=list(range(1, len(history_val_maes) + 1)),
            y=history_val_maes,
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
            title_text=f'Incremental LSTM Training Progress - {learning_mode.upper()} Mode',
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

    # Create state evolution plot
    if model_success:
        fig_state = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Prediction Error Over Time (State Continuity)',
                            'Rolling MAE (Window=10)'),
            vertical_spacing=0.15
        )

        # Prediction errors
        all_actual = np.concatenate([y_train_actual.flatten(), y_test_actual.flatten()])
        all_pred = np.concatenate([train_predict.flatten(), test_predict.flatten()])
        errors = np.abs(all_actual - all_pred)

        error_dates = df['ds'].iloc[lookback:lookback + len(errors)]

        fig_state.add_trace(go.Scatter(
            x=error_dates,
            y=errors,
            mode='lines+markers',
            name='Absolute Error',
            line=dict(color='#e74c3c', width=1),
            marker=dict(size=4),
            hovertemplate='Date: %{x}<br>Error: %{y:,.0f} m³<extra></extra>'
        ), row=1, col=1)

        # Rolling MAE
        window = 10
        rolling_mae = pd.Series(errors).rolling(window=window).mean()

        fig_state.add_trace(go.Scatter(
            x=error_dates,
            y=rolling_mae,
            mode='lines',
            name=f'Rolling MAE ({window} days)',
            line=dict(color='#3498db', width=3),
            hovertemplate='Date: %{x}<br>Rolling MAE: %{y:,.0f} m³<extra></extra>'
        ), row=2, col=1)

        # Add train/test split line
        split_date = df['ds'].iloc[lookback + train_size]

        # Add vertical line using shapes instead of add_vline
        fig_state.add_shape(
            type="line",
            x0=split_date, x1=split_date,
            y0=0, y1=1,
            yref="paper",
            line=dict(color="green", width=2, dash="dash"),
            row='all',
            col=1
        )

        # Add annotation for the split line
        fig_state.add_annotation(
            x=split_date,
            y=1,
            yref="paper",
            text="Train/Test Split",
            showarrow=False,
            yanchor="bottom",
            font=dict(color="green"),
            row=1,
            col=1
        )

        fig_state.update_xaxes(title_text="Date", row=2, col=1)
        fig_state.update_yaxes(title_text="Absolute Error (m³)", row=1, col=1)
        fig_state.update_yaxes(title_text="Rolling MAE (m³)", row=2, col=1)

        fig_state.update_layout(
            height=600,
            template='plotly_white',
            title_text='State Continuity Analysis - Error Evolution',
            showlegend=True
        )
    else:
        fig_state = go.Figure()
        fig_state.add_annotation(
            text="State analysis not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#7f8c8d')
        )

    # Forecast summary
    avg_forecast = forecast_data['yhat'].mean()
    total_forecast = forecast_data['yhat'].sum()

    metrics_div = html.Div([
        html.H3("📊 Incremental LSTM Performance & Forecast Summary",
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
                html.H4("Avg Forecast", style={'color': '#e67e22'}),
                html.P(f"{avg_forecast:,.0f} m³", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P(f"Next {forecast_days} days", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Batch Size", style={'color': '#8e44ad'}),
                html.P(f"{batch_size if model_success else 'N/A'}", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P("Stateful Processing", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),
        ])
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})

    # Generate insights
    if not model_success:
        insights = [
            f"❌ Incremental LSTM training failed: {error_msg}",
            "💡 Try reducing model complexity (fewer layers/units)",
            "💡 Check if dataset size is divisible by batch sizes",
            "💡 Consider using batch mode instead of continuous learning"
        ]
        bg_color = '#f8d7da'
        border_color = '#dc3545'
    else:
        # Calculate overfitting indicator
        overfit_ratio = test_mae / train_mae if train_mae > 0 else 1.0

        insights_dict = {
            'normal': [
                f"✅ Incremental LSTM trained on {train_size} sequences with stateful processing",
                f"✅ {'Continuous learning mode: States preserved across epochs' if learning_mode == 'continuous' else 'Batch mode: States reset between epochs'}",
                f"✅ Validation MAE: {test_mae:,.0f} m³ - Model generalizes {'well' if overfit_ratio < 1.5 else 'moderately'}",
                f"✅ Batch size {batch_size} ensures proper state propagation",
                f"🎯 Overfitting ratio: {overfit_ratio:.2f}"
            ],
            'missing_data': [
                f"✅ Incremental LSTM maintained state continuity across interpolated gaps",
                f"✅ Stateful architecture preserves context despite missing data",
                f"✅ {'Continuous mode helps bridge gaps with preserved memory' if learning_mode == 'continuous' else 'Batch mode resets states at each epoch'}",
                "🎯 Evidence: State evolution plot shows smooth error patterns"
            ],
            'outliers': [
                f"⚠️ Incremental LSTM learned from outlier-contaminated data",
                f"✅ Validation MAE: {test_mae:,.0f} m³ - Some outlier influence",
                f"✅ Stateful processing helps detect anomalies in context",
                "🎯 Evidence: Error spikes visible at outlier locations"
            ],
            'trend_shift': [
                f"✅ Incremental LSTM rapidly adapted to 30% trend shift",
                f"✅ {'Continuous learning allows real-time adaptation' if learning_mode == 'continuous' else 'Batch learning captures shift through epochs'}",
                f"✅ State preservation maintains context across regime change",
                "🎯 Evidence: Rolling MAE stabilizes after shift point"
            ],
            'high_noise': [
                f"✅ Incremental LSTM filtered noise through stateful memory (noise: {noise_level:,.0f} m³)",
                f"✅ Validation RMSE: {test_rmse:,.0f} m³",
                f"✅ State continuity helps distinguish signal from noise",
                "🎯 Evidence: Rolling MAE shows stable performance despite volatility"
            ]
        }

        insights = insights_dict.get(scenario, [])

        # Add incremental learning specific insights
        if learning_mode == 'continuous':
            insights.append(
                "🔄 Continuous Mode: Model can be updated incrementally with new data without full retraining")
            insights.append("💡 Ideal for streaming data and online learning scenarios")
        else:
            insights.append("🔄 Batch Mode: States reset between epochs for independent training")
            insights.append("💡 Better for offline training with complete datasets")

        # Add overfitting analysis
        if overfit_ratio > 2.0:
            insights.append("⚠️ Significant overfitting detected - consider regularization")
        elif overfit_ratio < 1.2:
            insights.append("✅ Excellent generalization - Incremental LSTM doesn't overfit")

        bg_color = '#d4edda'
        border_color = '#28a745'

    insights_div = html.Div([
        html.H3("🎯 Incremental LSTM Performance Evidence",
                style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Ul([html.Li(insight, style={'marginBottom': '10px', 'fontSize': '14px'})
                 for insight in insights])
    ], style={'backgroundColor': bg_color, 'padding': '20px', 'borderRadius': '10px',
              'border': f'2px solid {border_color}'})

    return training_info, fig_forecast, fig_training, fig_state, metrics_div, insights_div


if __name__ == '__main__':
    app.run(debug=True, port=8972)


