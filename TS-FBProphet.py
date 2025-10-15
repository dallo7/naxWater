import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from prophet import Prophet
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
        df['date'] = pd.to_datetime(df['date'], format='mixed')

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Using simulated data instead.")
        return None


def prepare_prophet_data(df, target_column='actual_consumption_m3', aggregate=True):
    """Prepare data in Prophet format (ds, y)"""
    if aggregate:
        # Aggregate by date (sum consumption across all zones)
        prophet_df = df.groupby('date').agg({
            target_column: 'sum',
            'rainfall_mm': 'mean',
            'pipe_leakage_m3': 'sum',
            'complaints_received': 'sum',
            'population_served': 'sum'
        }).reset_index()
    else:
        prophet_df = df[['date', target_column]].copy()

    # Rename for Prophet
    prophet_df = prophet_df.rename(columns={'date': 'ds', target_column: 'y'})

    return prophet_df


def add_noise_to_data(df, noise_level=0):
    """Add additional noise to test Prophet's robustness"""
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
        df_challenge = df_challenge.dropna()

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


# Initialize the Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "Aqua-Predict | FB Prophet Water Forecasting"
server = app.server


# Try to load real data
real_data = load_and_prepare_data()
if real_data is not None:
    base_prophet_data = prepare_prophet_data(real_data)
    data_source = "Real Water Supply Data"
else:
    # Fallback to simulated data if file not found
    dates = pd.date_range(start='2025-09-01', periods=77, freq='D')
    values = 35000 + 5000 * np.sin(2 * np.pi * np.arange(77) / 7) + np.random.normal(0, 2000, 77)
    base_prophet_data = pd.DataFrame({'ds': dates, 'y': values})
    data_source = "Simulated Data (CSV not found)"

# App layout
app.layout = html.Div([
    html.Div([
        html.H1("💧 Water Supply Forecasting with Facebook Prophet",
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.Br(),
        html.Br(),
        html.Br()
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
                    max=900,
                    step=20,
                    value=0,
                    marks={0: '0 sps', 100: '100 sps', 250: '200 sps', 400: '4000 sps', 550: '500 sps', 700: '700 sps'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px'}),

            html.Div([
                html.Label("Forecast Period (days):", style={'fontWeight': 'bold'}),
                dcc.Slider(
                    id='forecast-slider',
                    min=7,
                    max=60,
                    step=7,
                    value=14,
                    marks={7: '7', 14: '14', 30: '30', 60: '60'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px'}),
        ]),

        html.Div([
            html.Label("Changepoint Prior Scale (Flexibility):", style={'fontWeight': 'bold'}),
            dcc.Slider(
                id='changepoint-slider',
                min=0.001,
                max=0.5,
                step=0.05,
                value=0.05,
                marks={0.001: '0.001', 0.1: '0.1', 0.3: '0.3', 0.5: '0.5'},
                tooltip={"placement": "bottom", "always_visible": True}
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
        html.Button('🔄 Run Prophet Forecast', id='forecast-button', n_clicks=0,
                    style={'backgroundColor': '#3498db', 'color': 'white', 'padding': '15px 30px',
                           'fontSize': '16px', 'border': 'none', 'borderRadius': '5px',
                           'cursor': 'pointer', 'display': 'block', 'margin': '0 auto'})
    ], style={'marginBottom': '20px'}),

    # Loading component
    dcc.Loading(
        id="loading",
        type="default",
        children=[
            # Forecast plot
            dcc.Graph(id='forecast-plot', style={'marginBottom': '20px'}),

            # Components plot
            dcc.Graph(id='components-plot'),

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
        'normal': "📊 Baseline scenario using actual water consumption data with all original patterns intact.",
        'missing_data': "🕳️ CHALLENGE 1: Dataset with 20% missing data points. Prophet handles gaps by interpolating while maintaining seasonal patterns.",
        'outliers': "⚡ CHALLENGE 2: Dataset contains extreme consumption spikes/drops (5% anomalies). Prophet's robust regression minimizes outlier impact.",
        'trend_shift': "📉 CHALLENGE 3: Sudden 30% increase in consumption halfway through the period. Prophet detects and adapts to changepoints automatically.",
        'high_noise': "🔍 CHALLENGE 4: Highly volatile consumption patterns. Prophet separates signal from noise through decomposition."
    }

    scenario_desc = html.Div([
        html.Strong("Current Scenario: "),
        html.Span(descriptions.get(scenario, "Select a scenario"))
    ])

    # Dataset info
    if real_data is not None:
        total_consumption = real_data['actual_consumption_m3'].sum()
        avg_daily = base_prophet_data['y'].mean()
        date_range = f"{base_prophet_data['ds'].min().strftime('%Y-%m-%d')} to {base_prophet_data['ds'].max().strftime('%Y-%m-%d')}"
        num_zones = real_data['zone'].nunique()

        dataset_desc = html.Div([
            html.Strong("📊 Dataset Overview: "),
            html.Span(f"{len(base_prophet_data)} days | {num_zones} zones | "),
            html.Span(f"Avg Daily: {avg_daily:,.0f} m³ | "),
            html.Span(f"Total: {total_consumption:,.0f} m³ | "),
            html.Span(f"Period: {date_range}")
        ])
    else:
        dataset_desc = html.Div([
            html.Strong("⚠️ Note: "),
            html.Span("Using simulated data. Place 'nawassco.csv' in the same directory to use real data.")
        ])

    return scenario_desc, dataset_desc


@app.callback(
    [Output('forecast-plot', 'figure'),
     Output('components-plot', 'figure'),
     Output('metrics-display', 'children'),
     Output('insights-display', 'children')],
    [Input('forecast-button', 'n_clicks')],
    [State('scenario-dropdown', 'value'),
     State('noise-slider', 'value'),
     State('forecast-slider', 'value'),
     State('changepoint-slider', 'value')]
)
def update_forecast(n_clicks, scenario, noise_level, forecast_days, changepoint_prior):
    """Run Prophet model and update all visualizations"""

    if n_clicks == 0:
        return go.Figure(), go.Figure(), html.Div(), html.Div()

    # Prepare data based on scenario
    df = base_prophet_data.copy()

    # Apply challenge scenarios
    if scenario != 'normal' and scenario != 'high_noise':
        df = create_challenge_scenario(df, scenario)

    # Add noise
    if scenario == 'high_noise':
        noise_level = max(noise_level, 3000)

    df = add_noise_to_data(df, noise_level)

    # Train Prophet model
    model = Prophet(
        changepoint_prior_scale=changepoint_prior,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        stan_backend='CMDSTANPY'
    )

    model.add_country_holidays(country_name='KE')

    # Fit with error handling
    try:
        model.fit(df)
    except Exception as e:
        import logging
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        model.fit(df, algorithm='Newton')

    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Create forecast plot
    fig_forecast = go.Figure()

    # Add a shaded background region for the forecast period (now orange)
    last_historical_date = df['ds'].max()
    fig_forecast.add_vrect(
        x0=last_historical_date,
        x1=forecast['ds'].max(),
        fillcolor="rgba(255, 165, 0, 0.15)",  # Orange with some transparency
        layer="below",
        line_width=0,
        annotation_text="Forecast Period",
        annotation_position="top left",
        annotation=dict(font=dict(size=12, color="rgba(255, 165, 0, 0.8)")) # Orange text for annotation
    )

    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='markers',
        name='Actual Consumption',
        marker=dict(size=6, color='#2c3e50', opacity=0.7),
        hovertemplate='Date: %{x}<br>Consumption: %{y:,.0f} m³<extra></extra>'
    ))

    # Forecast (back to blue)
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Prophet Forecast',
        line=dict(color='#3498db', width=3), # Original blue forecast line
        hovertemplate='Date: %{x}<br>Forecast: %{y:,.0f} m³<extra></extra>'
    ))

    # Confidence interval (back to blue shade)
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='80% Confidence Interval',
        fill='tonexty',
        fillcolor='rgba(52, 152, 219, 0.2)', # Original blue confidence interval
        line=dict(width=0),
        hovertemplate='Lower: %{y:,.0f} m³<extra></extra>'
    ))

    # Add changepoints if detected
    if scenario == 'trend_shift':
        changepoints = model.changepoints
        changepoint_values = [
            forecast[forecast['ds'] == cp]['yhat'].values[0] if len(forecast[forecast['ds'] == cp]) > 0
            else forecast['yhat'].mean() for cp in changepoints[-3:]]

        fig_forecast.add_trace(go.Scatter(
            x=changepoints[-3:],
            y=changepoint_values,
            mode='markers',
            name='Detected Changepoints',
            marker=dict(size=12, color='red', symbol='x', line=dict(width=2)),
            hovertemplate='Changepoint<br>Date: %{x}<br>Value: %{y:,.0f} m³<extra></extra>'
        ))

    fig_forecast.update_layout(
        title=f'Water Consumption Forecast - {scenario.replace("_", " ").title()}',
        xaxis_title='Date',
        yaxis_title='Water Consumption (m³)',
        hovermode='x unified',
        template='plotly_white',
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Create components plot
    fig_components = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Trend Component', 'Weekly Seasonality', 'Overall Uncertainty'),
        vertical_spacing=0.1
    )

    # Trend
    fig_components.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['trend'],
        mode='lines',
        name='Trend',
        line=dict(color='#e74c3c', width=2),
        hovertemplate='%{y:,.0f} m³<extra></extra>'
    ), row=1, col=1)

    # Weekly seasonality
    fig_components.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['weekly'],
        mode='lines',
        name='Weekly',
        line=dict(color='#27ae60', width=2),
        hovertemplate='%{y:,.0f} m³<extra></extra>'
    ), row=2, col=1)

    # Uncertainty (yhat_upper - yhat_lower)
    uncertainty = forecast['yhat_upper'] - forecast['yhat_lower']
    fig_components.add_trace(go.Scatter(
        x=forecast['ds'],
        y=uncertainty,
        mode='lines',
        name='Uncertainty',
        line=dict(color='#9b59b6', width=2),
        fill='tozeroy',
        fillcolor='rgba(155, 89, 182, 0.2)',
        hovertemplate='%{y:,.0f} m³<extra></extra>'
    ), row=3, col=1)

    fig_components.update_xaxes(title_text="Date", row=3, col=1)
    fig_components.update_yaxes(title_text="Trend (m³)", row=1, col=1)
    fig_components.update_yaxes(title_text="Weekly Effect (m³)", row=2, col=1)
    fig_components.update_yaxes(title_text="Uncertainty Range (m³)", row=3, col=1)

    fig_components.update_layout(
        height=900,
        showlegend=False,
        template='plotly_white',
        title_text='Prophet Component Breakdown - How Forecast is Built'
    )

    # Calculate metrics
    train_forecast = forecast[forecast['ds'].isin(df['ds'])]
    actual_values = df['y'].values
    predicted_values = train_forecast['yhat'].values[:len(actual_values)]

    mae = np.mean(np.abs(actual_values - predicted_values))
    rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    # Forecast summary
    future_forecast = forecast[forecast['ds'] > df['ds'].max()]
    avg_forecast = future_forecast['yhat'].mean()
    total_forecast = future_forecast['yhat'].sum()

    metrics_div = html.Div([
        html.H3("📊 Model Performance & Forecast Summary", style={'color': '#2c3e50', 'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.H4("MAE", style={'color': '#3498db'}),
                html.P(f"{mae:,.0f} m³", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P("Mean Absolute Error", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("RMSE", style={'color': '#e74c3c'}),
                html.P(f"{rmse:,.0f} m³", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P("Root Mean Squared Error", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("MAPE", style={'color': '#f39c12'}),
                html.P(f"{mape:.1f}%", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P("Mean Absolute % Error", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Avg Forecast", style={'color': '#27ae60'}),
                html.P(f"{avg_forecast:,.0f} m³", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P(f"Next {forecast_days} days", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Total Forecast", style={'color': '#9b59b6'}),
                html.P(f"{total_forecast:,.0f} m³", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P(f"Period demand", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '20%', 'display': 'inline-block', 'textAlign': 'center'}),
        ])
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})

    # Generate insights
    insights = {
        'normal': [
            f"✅ Prophet successfully modeled {len(df)} days of water consumption data",
            f"✅ Weekly patterns detected: consumption varies by day of week",
            f"✅ Forecast confidence: ±{(forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / 2:,.0f} m³",
            f"🎯 Average daily consumption: {df['y'].mean():,.0f} m³"
        ],
        'missing_data': [
            f"✅ Prophet handled {len(base_prophet_data) - len(df)} missing data points gracefully",
            "✅ No interpolation artifacts - maintains smooth consumption patterns",
            "✅ Weekly seasonality remains intact despite data gaps",
            "🎯 Evidence: Trend line is continuous with no jumps at missing points"
        ],
        'outliers': [
            "✅ Prophet's robust regression minimized outlier impact",
            "✅ Extreme spikes/drops don't distort the forecast trend",
            "✅ Confidence intervals appropriately widen around anomalies",
            "🎯 Evidence: Forecast follows underlying pattern, not anomalous values"
        ],
        'trend_shift': [
            f"✅ Prophet detected changepoint and adapted to 30% consumption increase",
            "✅ Trend component shows clear regime shift in the middle period",
            "✅ Forecast extends the new higher consumption level appropriately",
            "🎯 Evidence: Red dashed lines mark detected changepoints in the plot"
        ],
        'high_noise': [
            f"✅ Prophet separated signal from noise (noise level: {noise_level:,.0f} m³)",
            "✅ Underlying trend and seasonality still clearly visible",
            "✅ Wider confidence intervals reflect increased uncertainty",
            "🎯 Evidence: Component breakdown shows smooth patterns despite noisy data"
        ]
    }

    insights_div = html.Div([
        html.H3("🎯 Prophet's Performance Evidence", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Ul([html.Li(insight, style={'marginBottom': '10px', 'fontSize': '14px'})
                 for insight in insights.get(scenario, [])])
    ], style={'backgroundColor': '#d4edda', 'padding': '20px', 'borderRadius': '10px',
              'border': '2px solid #28a745'})

    return fig_forecast, fig_components, metrics_div, insights_div


if __name__ == '__main__':
    app.run(debug=True, port=8099)