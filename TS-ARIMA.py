import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
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
    """Prepare data for ARIMA (ds, y)"""
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
    """Add additional noise to test ARIMA's robustness"""
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
        # ARIMA requires complete data - interpolate
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


def check_stationarity(timeseries):
    """Check if time series is stationary using ADF test"""
    result = adfuller(timeseries.dropna())
    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < 0.05
    }


# Initialize the Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "Aqua-Predict | ARIMA"
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
        html.H1("💧 Water Supply Forecasting with ARIMA",
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
                    max=60,
                    step=7,
                    value=14,
                    marks={7: '7', 14: '14', 30: '30', 60: '60'},
                    tooltip={"placement": "bottom", "always_visible": True}
                )
            ], style={'width': '45%', 'display': 'inline-block', 'padding': '20px'}),
        ]),

        html.Div([
            html.Div([
                html.Label("ARIMA Order (p,d,q):", style={'fontWeight': 'bold'}),
                html.Div([
                    dcc.Input(id='p-order', type='number', value=2, min=0, max=5, step=1,
                              style={'width': '30%', 'marginRight': '5px'}),
                    dcc.Input(id='d-order', type='number', value=1, min=0, max=2, step=1,
                              style={'width': '30%', 'marginRight': '5px'}),
                    dcc.Input(id='q-order', type='number', value=2, min=0, max=5, step=1,
                              style={'width': '30%'}),
                ], style={'display': 'flex', 'justifyContent': 'space-between'}),
                html.P("p=AR order, d=Differencing, q=MA order",
                       style={'fontSize': '11px', 'color': '#7f8c8d', 'marginTop': '5px'})
            ], style={'width': '90%', 'margin': '0 auto', 'padding': '20px'}),
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
        html.Button('🔄 Run ARIMA Forecast', id='forecast-button', n_clicks=0,
                    style={'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '15px 30px',
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
        'normal': "📊 Baseline scenario using actual water consumption data with ARIMA modeling.",
        'missing_data': "🕳️ CHALLENGE 1: Dataset with 20% missing data points. ARIMA requires complete data - interpolation applied.",
        'outliers': "⚡ CHALLENGE 2: Dataset contains extreme consumption spikes/drops (5% anomalies). ARIMA may be sensitive to outliers.",
        'trend_shift': "📉 CHALLENGE 3: Sudden 30% increase in consumption halfway through. ARIMA will attempt to model the structural break.",
        'high_noise': "🔍 CHALLENGE 4: Highly volatile consumption patterns. ARIMA's MA component helps smooth noise."
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
    [Output('forecast-plot', 'figure'),
     Output('components-plot', 'figure'),
     Output('metrics-display', 'children'),
     Output('insights-display', 'children')],
    [Input('forecast-button', 'n_clicks')],
    [State('scenario-dropdown', 'value'),
     State('noise-slider', 'value'),
     State('forecast-slider', 'value'),
     State('p-order', 'value'),
     State('d-order', 'value'),
     State('q-order', 'value')]
)
def update_forecast(n_clicks, scenario, noise_level, forecast_days, p_order, d_order, q_order):
    """Run ARIMA model and update all visualizations"""

    if n_clicks == 0:
        return go.Figure(), go.Figure(), html.Div(), html.Div()

    # Prepare data based on scenario
    df = base_ts_data.copy()

    # Apply challenge scenarios
    if scenario != 'normal' and scenario != 'high_noise':
        df = create_challenge_scenario(df, scenario)

    # Add noise
    if scenario == 'high_noise':
        noise_level = max(noise_level, 3000)

    df = add_noise_to_data(df, noise_level)

    # Check stationarity
    stationarity_test = check_stationarity(df['y'])

    # Fit ARIMA model
    try:
        model = ARIMA(df['y'], order=(p_order, d_order, q_order))
        fitted_model = model.fit()

        # Make predictions
        forecast_result = fitted_model.forecast(steps=forecast_days)

        # Get confidence intervals
        forecast_df = fitted_model.get_forecast(steps=forecast_days)
        forecast_ci = forecast_df.conf_int()

        # Prepare forecast dataframe
        forecast_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1),
                                       periods=forecast_days, freq='D')

        forecast_data = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast_result.values,
            'yhat_lower': forecast_ci.iloc[:, 0].values,
            'yhat_upper': forecast_ci.iloc[:, 1].values
        })

        # Get fitted values
        fitted_values = fitted_model.fittedvalues

        model_success = True
        error_msg = None

    except Exception as e:
        model_success = False
        error_msg = str(e)
        # Create empty forecast for error case
        forecast_dates = pd.date_range(start=df['ds'].iloc[-1] + timedelta(days=1),
                                       periods=forecast_days, freq='D')
        forecast_data = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': [df['y'].mean()] * forecast_days,
            'yhat_lower': [df['y'].mean() * 0.9] * forecast_days,
            'yhat_upper': [df['y'].mean() * 1.1] * forecast_days
        })
        fitted_values = pd.Series([df['y'].mean()] * len(df))

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
        # Fitted values
        fig_forecast.add_trace(go.Scatter(
            x=df['ds'],
            y=fitted_values,
            mode='lines',
            name='ARIMA Fitted',
            line=dict(color='#27ae60', width=2),
            hovertemplate='Date: %{x}<br>Fitted: %{y:,.0f} m³<extra></extra>'
        ))

    # Forecast
    fig_forecast.add_trace(go.Scatter(
        x=forecast_data['ds'],
        y=forecast_data['yhat'],
        mode='lines',
        name='ARIMA Forecast',
        line=dict(color='#e74c3c', width=3),
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
        fillcolor='rgba(231, 76, 60, 0.2)',
        line=dict(width=0),
        hovertemplate='Lower: %{y:,.0f} m³<extra></extra>'
    ))

    fig_forecast.update_layout(
        title=f'Water Consumption Forecast - ARIMA({p_order},{d_order},{q_order}) - {scenario.replace("_", " ").title()}',
        xaxis_title='Date',
        yaxis_title='Water Consumption (m³)',
        hovermode='x unified',
        template='plotly_white',
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Create components plot using seasonal decomposition
    if len(df) >= 14:  # Need enough data for decomposition
        try:
            decomposition = seasonal_decompose(df.set_index('ds')['y'],
                                               model='additive',
                                               period=7,
                                               extrapolate_trend='freq')

            fig_components = make_subplots(
                rows=4, cols=1,
                subplot_titles=('Original Series', 'Trend Component',
                                'Seasonal Component', 'Residual Component'),
                vertical_spacing=0.08
            )

            # Original
            fig_components.add_trace(go.Scatter(
                x=df['ds'],
                y=df['y'],
                mode='lines',
                name='Original',
                line=dict(color='#2c3e50', width=2)
            ), row=1, col=1)

            # Trend
            fig_components.add_trace(go.Scatter(
                x=df['ds'],
                y=decomposition.trend,
                mode='lines',
                name='Trend',
                line=dict(color='#e74c3c', width=2)
            ), row=2, col=1)

            # Seasonal
            fig_components.add_trace(go.Scatter(
                x=df['ds'],
                y=decomposition.seasonal,
                mode='lines',
                name='Seasonal',
                line=dict(color='#27ae60', width=2)
            ), row=3, col=1)

            # Residual
            fig_components.add_trace(go.Scatter(
                x=df['ds'],
                y=decomposition.resid,
                mode='lines',
                name='Residual',
                line=dict(color='#9b59b6', width=2)
            ), row=4, col=1)

            fig_components.update_xaxes(title_text="Date", row=4, col=1)
            fig_components.update_yaxes(title_text="Value (m³)", row=1, col=1)
            fig_components.update_yaxes(title_text="Trend (m³)", row=2, col=1)
            fig_components.update_yaxes(title_text="Seasonal (m³)", row=3, col=1)
            fig_components.update_yaxes(title_text="Residual (m³)", row=4, col=1)

            fig_components.update_layout(
                height=1000,
                showlegend=False,
                template='plotly_white',
                title_text='Time Series Decomposition - ARIMA Component Analysis'
            )
        except:
            fig_components = go.Figure()
            fig_components.add_annotation(
                text="Insufficient data for decomposition",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    else:
        fig_components = go.Figure()
        fig_components.add_annotation(
            text="Need at least 14 days for decomposition",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

    # Calculate metrics
    if model_success:
        residuals = df['y'].values - fitted_values.values
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        mape = np.mean(np.abs(residuals / df['y'].values)) * 100

        aic = fitted_model.aic
        bic = fitted_model.bic
    else:
        mae = rmse = mape = aic = bic = 0

    # Forecast summary
    avg_forecast = forecast_data['yhat'].mean()
    total_forecast = forecast_data['yhat'].sum()

    metrics_div = html.Div([
        html.H3("📊 ARIMA Model Performance & Forecast Summary",
                style={'color': '#2c3e50', 'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.H4("MAE", style={'color': '#e74c3c'}),
                html.P(f"{mae:,.0f} m³" if model_success else "N/A",
                       style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P("Mean Absolute Error", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("RMSE", style={'color': '#c0392b'}),
                html.P(f"{rmse:,.0f} m³" if model_success else "N/A",
                       style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P("Root Mean Squared Error", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("AIC", style={'color': '#f39c12'}),
                html.P(f"{aic:,.0f}" if model_success else "N/A",
                       style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P("Akaike Info Criterion", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("BIC", style={'color': '#d35400'}),
                html.P(f"{bic:,.0f}" if model_success else "N/A",
                       style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P("Bayesian Info Criterion", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Avg Forecast", style={'color': '#27ae60'}),
                html.P(f"{avg_forecast:,.0f} m³", style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P(f"Next {forecast_days} days", style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),

            html.Div([
                html.H4("Stationarity", style={'color': '#8e44ad'}),
                html.P("✓" if stationarity_test['is_stationary'] else "✗",
                       style={'fontSize': '20px', 'fontWeight': 'bold'}),
                html.P(f"p-value: {stationarity_test['p_value']:.4f}",
                       style={'fontSize': '11px', 'color': '#7f8c8d'})
            ], style={'width': '16%', 'display': 'inline-block', 'textAlign': 'center'}),
        ])
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px'})

    # Generate insights
    if not model_success:
        insights = [
            f"❌ ARIMA model failed to converge: {error_msg}",
            "💡 Try adjusting the (p,d,q) parameters",
            "💡 This scenario may be too challenging for ARIMA",
            "💡 Consider data preprocessing or alternative models"
        ]
        bg_color = '#f8d7da'
        border_color = '#dc3545'
    else:
        insights_dict = {
            'normal': [
                f"✅ ARIMA({p_order},{d_order},{q_order}) successfully modeled {len(df)} days of water consumption",
                f"✅ Series is {'stationary' if stationarity_test['is_stationary'] else 'non-stationary (differencing applied)'}",
                f"✅ AIC: {aic:.0f}, BIC: {bic:.0f} - Lower values indicate better fit",
                f"🎯 Average daily consumption: {df['y'].mean():,.0f} m³"
            ],
            'missing_data': [
                f"✅ ARIMA handled interpolated missing data (originally 20% gaps)",
                "⚠️ ARIMA requires complete data - gaps were filled via linear interpolation",
                f"✅ Model maintains {'stationarity' if stationarity_test['is_stationary'] else 'handles non-stationarity with d={d_order}'}",
                "🎯 Evidence: Smooth fitted line despite original data gaps"
            ],
            'outliers': [
                "⚠️ ARIMA is sensitive to outliers - performance may degrade",
                "✅ Moving average component (q) helps smooth some outlier effects",
                f"⚠️ RMSE ({rmse:,.0f}) may be inflated due to anomalies",
                "🎯 Evidence: Fitted values may deviate significantly at outlier points"
            ],
            'trend_shift': [
                f"⚠️ ARIMA struggles with structural breaks (sudden 30% shift)",
                f"✅ Higher differencing (d={d_order}) helps adapt to changing levels",
                "⚠️ May need regime-switching models for better performance",
                "🎯 Evidence: Forecast may under/over-estimate after the break point"
            ],
            'high_noise': [
                f"✅ ARIMA's MA component (q={q_order}) helps filter noise",
                f"✅ Model identifies signal despite volatility (noise: {noise_level:,.0f} m³)",
                "✅ Wider confidence intervals appropriately reflect uncertainty",
                "🎯 Evidence: Fitted line smooths noisy observations effectively"
            ]
        }
        insights = insights_dict.get(scenario, [])
        bg_color = '#d4edda'
        border_color = '#28a745'

    insights_div = html.Div([
        html.H3("🎯 ARIMA Performance Evidence",
                style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Ul([html.Li(insight, style={'marginBottom': '10px', 'fontSize': '14px'})
                 for insight in insights])
    ], style={'backgroundColor': bg_color, 'padding': '20px', 'borderRadius': '10px',
              'border': f'2px solid {border_color}'})

    return fig_forecast, fig_components, metrics_div, insights_div


if __name__ == '__main__':

    app.run(debug=True, port=8051)
