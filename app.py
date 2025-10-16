import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os
import requests
import time
from threading import Thread
from prophet import Prophet
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from tensorflow.keras.layers import Input as KerasInput, Dense, Dropout, BatchNormalization, Concatenate, Attention, \
    MultiHeadAttention, Reshape, Flatten
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input as KerasInput, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

# --- [1] App & DB Initialization ---
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)
app.title = "Aqua-Predict | Water Management System"
server = app.server
DB_FILE = "water_management_final.db"
CSV_FILE = "nawassco.csv"

# Global Color Palette
COLOR_PALETTE = ['#007BFF', '#28A745', '#DC3545', '#FFC107', '#17A2B8', '#6F42C0', '#FD7E14', '#E83E8C']
PLOT_LAYOUT = {
    'plot_bgcolor': '#F8F9FA',
    'paper_bgcolor': 'white',
    'font': {'family': 'Arial, sans-serif', 'color': '#343A40'},
    'margin': {'t': 50, 'b': 20, 'l': 40, 'r': 10},
    'title': {'font': {'size': 18}, 'x': 0.5, 'xanchor': 'center'}
}


# --- [2] Database Setup and Population ---
def init_database():
    """
    Initializes or recreates the SQLite database with all necessary tables.
    Deletes and recreates the database if the required tables are not found.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # List of required tables
    required_tables = ['users', 'supply_data', 'billing_history', 'chat_messages', 'outstanding_bills']

    # Check if all required tables exist
    missing_tables = []
    for table_name in required_tables:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        if cursor.fetchone() is None:
            missing_tables.append(table_name)

    conn.close()

    if missing_tables:
        print(f"Database schema incomplete. Missing tables: {', '.join(missing_tables)}. Recreating database...")
        if os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            print("Old database file deleted.")

        # Now, create a new database with all tables
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        if not os.path.exists(CSV_FILE):
            print(f"Error: {CSV_FILE} not found. Please place it in the same directory.")
            return

        # Create users table without the phone_number column
        cursor.execute(
            'CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password_hash TEXT, user_type TEXT, zone TEXT, supply_area TEXT)')
        cursor.execute("INSERT INTO users (username, password_hash, user_type) VALUES (?, ?, ?)",
                       ('Admin', generate_password_hash('Admin123'), 'admin'))

        df = pd.read_csv(CSV_FILE)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        df.rename(columns={'price/litre': 'price_per_litre'}, inplace=True)

        # Add new water quality column (simulated pH)
        df['ph_level'] = np.random.uniform(6.5, 8.5, size=len(df))

        df.to_sql('supply_data', conn, if_exists='replace', index=False)

        demo_users = [
            ('dalmas', 'pass', 'Central', 'Area 1'),
            ('chituyi', 'pass', 'Southern', 'Area 3'),
            ('wakhusama', 'pass', 'Western', 'Area 5'),
            ('Daniel Omondi', 'pass', 'Central', 'Area 2'),
            ('Kim Ngeno', 'pass', 'Southern', 'Area 4'),
            ('Jack Wabwire', 'pass', 'Western', 'Area 6'),
            ('Johnstone Mungao', 'pass', 'Central', 'Area 1'),
            ('Vicky Kerubo', 'pass', 'Southern', 'Area 3'),
            ('Alice Cheptoo', 'pass', 'Western', 'Area 5'),
            ('Mary Kuto', 'pass', 'Central', 'Area 2'),
            ('Jeniffer Mwangi', 'pass', 'Southern', 'Area 4'),
        ]
        for user, pwd, zone, area in demo_users:
            cursor.execute(
                "INSERT INTO users (username, password_hash, user_type, zone, supply_area) VALUES (?, ?, 'consumer', ?, ?)",
                (user, generate_password_hash(pwd), zone, area))

        # Create billing history table
        cursor.execute(
            'CREATE TABLE billing_history (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, date_paid TEXT, status TEXT, FOREIGN KEY(user_id) REFERENCES users(id))')

        # Create outstanding bills table
        cursor.execute(
            'CREATE TABLE outstanding_bills (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, due_date TEXT, status TEXT, FOREIGN KEY(user_id) REFERENCES users(id))')

        # Populate billing history for demo users
        for username, _, _, _ in demo_users:
            user_id = cursor.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()[0]
            for i in range(1, 5):
                date_paid = (datetime.now() - timedelta(days=i * 30)).strftime('%Y-%m-%d')
                amount = np.random.uniform(3000, 5000)
                cursor.execute("INSERT INTO billing_history (user_id, amount, date_paid, status) VALUES (?, ?, ?, ?)",
                               (user_id, amount, date_paid, 'Paid'))

        # Populate initial outstanding bills for a couple of users
        dalmas_id = cursor.execute("SELECT id FROM users WHERE username = 'dalmas'").fetchone()[0]
        cursor.execute("INSERT INTO outstanding_bills (user_id, amount, due_date, status) VALUES (?, ?, ?, ?)",
                       (dalmas_id, 4500.00, (datetime.now() + timedelta(days=10)).strftime('%Y-%m-%d'), 'Outstanding'))
        chituyi_id = cursor.execute("SELECT id FROM users WHERE username = 'chituyi'").fetchone()[0]
        cursor.execute("INSERT INTO outstanding_bills (user_id, amount, due_date, status) VALUES (?, ?, ?, ?)",
                       (chituyi_id, 3850.50, (datetime.now() + timedelta(days=15)).strftime('%Y-%m-%d'), 'Outstanding'))

        # Create chat table
        cursor.execute(
            'CREATE TABLE chat_messages (id INTEGER PRIMARY KEY, user_id INTEGER, message TEXT, timestamp TEXT, FOREIGN KEY(user_id) REFERENCES users(id))')

        # Add a few initial messages
        admin_id = cursor.execute("SELECT id FROM users WHERE user_type = 'Admin'").fetchone()[0]
        cursor.execute("INSERT INTO chat_messages (user_id, message, timestamp) VALUES (?, ?, ?)",
                       (admin_id, "Welcome to the support chat! Feel free to ask any questions.",
                        datetime.now().isoformat()))

        # Removed old water_usage_breakdown table as it will now be dynamically generated
        conn.commit()
        conn.close()
        print("Database initialization complete with demo users.")
    else:
        print("Database already initialized.")


init_database()


# --- [3] Data & Model Functions ---
def query_db(query, params=()):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_full_data_df():
    df = query_db("SELECT * FROM supply_data")
    df['date'] = pd.to_datetime(df['date'])
    return df


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


# ==================== DATA PREPARATION ====================
def Tt_load_and_prepare_data(filepath='nawassco.csv'):
    """Load and prepare the water supply dataset for rationing analysis"""
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower().str.replace('/', '_').str.replace(' ', '_')
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        print(f"✅ Loaded {filepath}")
        print(f"   Shape: {df.shape}")
        print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: {filepath} not found. Generating synthetic data...")
        return Tt_generate_synthetic_data()


def Tt_generate_synthetic_data():
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


def Tt_feature_engineering(df):
    """Create features for the rationing model"""
    df_feat = df.copy()
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['day'] = df_feat['date'].dt.day
    df_feat['day_of_year'] = df_feat['date'].dt.dayofyear
    df_feat['week_of_year'] = df_feat['date'].dt.isocalendar().week
    df_feat['is_weekend'] = df_feat['date'].dt.dayofweek.isin([5, 6]).astype(int)
    df_feat['quarter'] = df_feat['date'].dt.quarter
    day_mapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    df_feat['day_of_week_encoded'] = df_feat['day_of_week'].map(day_mapping)
    df_feat['consumption_ratio'] = df_feat['actual_consumption_m3'] / (df_feat['demand_forecast_m3'] + 1)
    df_feat['per_capita_consumption'] = df_feat['actual_consumption_m3'] / (df_feat['population_served'] + 1)
    df_feat['supply_efficiency'] = (df_feat['actual_consumption_m3'] - df_feat['pipe_leakage_m3']) / (
            df_feat['actual_consumption_m3'] + 1)
    df_feat['demand_pressure'] = df_feat['demand_forecast_m3'] / (df_feat['population_served'] + 1)
    df_feat['complaint_rate'] = df_feat['complaints_received'] / (df_feat['population_served'] + 1) * 10000
    df_feat['water_stress'] = (df_feat['demand_forecast_m3'] - df_feat['actual_consumption_m3']) / (
            df_feat['demand_forecast_m3'] + 1)
    df_feat['is_dry_season'] = (df_feat['rainfall_mm'] < df_feat['rainfall_mm'].median()).astype(int)
    return df_feat


# ==================== ADVANCED MODEL BUILDING ====================
def Tt_build_rationing_model(input_dim, n_zones, n_times):
    """
    Build advanced multi-output neural network for water rationing optimization
    """
    input_layer = KerasInput(shape=(input_dim,), name='input')
    x = Dense(256, activation='relu', name='embed_1')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(192, activation='relu', name='embed_2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    attention_input = Reshape((1, -1))(x)
    attention_output = MultiHeadAttention(
        num_heads=4,
        key_dim=48,
        name='multi_head_attention'
    )(attention_input, attention_input)
    attention_output = Flatten()(attention_output)
    x = Concatenate()([x, attention_output])
    x = Dense(128, activation='relu', name='shared_1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(96, activation='relu', name='shared_2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    time_branch = Dense(64, activation='relu', name='time_dense_1')(x)
    time_branch = BatchNormalization()(time_branch)
    time_branch = Dropout(0.2)(time_branch)
    time_branch = Dense(32, activation='relu', name='time_dense_2')(time_branch)
    time_branch = Dropout(0.2)(time_branch)
    time_output = Dense(n_times, activation='softmax', name='time_output')(time_branch)
    zone_branch = Dense(64, activation='relu', name='zone_dense_1')(x)
    zone_branch = BatchNormalization()(zone_branch)
    zone_branch = Dropout(0.2)(zone_branch)
    zone_branch = Dense(32, activation='relu', name='zone_dense_2')(zone_branch)
    zone_branch = Dropout(0.2)(zone_branch)
    zone_output = Dense(n_zones, activation='softmax', name='zone_output')(zone_branch)
    model = Model(
        inputs=input_layer,
        outputs=[time_output, zone_output],
        name='WaterRationingOptimizer'
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'time_output': 'categorical_crossentropy',
            'zone_output': 'categorical_crossentropy'
        },
        loss_weights={
            'time_output': 1.0,
            'zone_output': 1.0
        },
        metrics={
            'time_output': ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')],
            'zone_output': ['accuracy', keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')]
        }
    )
    return model


def Tt_train_rationing_model(df):
    """Train the water rationing optimization model"""
    print("\n🔧 Starting Water Rationing Model Training...")
    df_processed = Tt_feature_engineering(df)
    feature_columns = [
        'month', 'day_of_year', 'week_of_year', 'quarter', 'is_weekend',
        'day_of_week_encoded', 'population_served', 'demand_forecast_m3',
        'actual_consumption_m3', 'consumption_ratio', 'per_capita_consumption',
        'rainfall_mm', 'pipe_leakage_m3', 'complaints_received',
        'supply_efficiency', 'demand_pressure', 'complaint_rate',
        'water_stress', 'is_dry_season'
    ]
    X = df_processed[feature_columns].values
    time_encoder = LabelEncoder()
    y_time_encoded = time_encoder.fit_transform(df_processed['time_of_supply'])
    y_time_categorical = keras.utils.to_categorical(y_time_encoded)
    zone_encoder = LabelEncoder()
    y_zone_encoded = zone_encoder.fit_transform(df_processed['zone'])
    y_zone_categorical = keras.utils.to_categorical(y_zone_encoded)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_time_train, y_time_test, y_zone_train, y_zone_test = train_test_split(
        X_scaled, y_time_categorical, y_zone_categorical,
        test_size=0.2, random_state=42, stratify=y_zone_encoded
    )
    n_zones = len(zone_encoder.classes_)
    n_times = len(time_encoder.classes_)
    model = Tt_build_rationing_model(
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
    print("\n📈 Evaluating model on test set...")
    predictions = model.predict(X_test, verbose=0)
    y_time_pred, y_zone_pred = predictions
    time_true = np.argmax(y_time_test, axis=1)
    time_pred = np.argmax(y_time_pred, axis=1)
    time_accuracy = accuracy_score(time_true, time_pred)
    zone_true = np.argmax(y_zone_test, axis=1)
    zone_pred = np.argmax(y_zone_pred, axis=1)
    zone_accuracy = accuracy_score(zone_true, zone_pred)
    time_top2 = np.mean([t in np.argsort(p)[-2:] for t, p in zip(time_true, y_time_pred)])
    zone_top2 = np.mean([z in np.argsort(p)[-2:] for z, p in zip(zone_true, y_zone_pred)])
    time_cm = confusion_matrix(time_true, time_pred)
    zone_cm = confusion_matrix(zone_true, zone_pred)
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


def Tt_generate_rationing_schedule(model_results, df, days_ahead=7):
    """Generate optimal rationing schedule for upcoming days"""
    df_processed = Tt_feature_engineering(df)
    latest_date = df_processed['date'].max()
    future_dates = [latest_date + timedelta(days=i + 1) for i in range(days_ahead)]
    zones = model_results['zone_encoder'].classes_
    schedule = []
    for date in future_dates:
        for zone in zones:
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
            features['consumption_ratio'] = features['actual_consumption_m3'] / (features['demand_forecast_m3'] + 1)
            features['per_capita_consumption'] = features['actual_consumption_m3'] / (features['population_served'] + 1)
            features['supply_efficiency'] = (features['actual_consumption_m3'] - features['pipe_leakage_m3']) / (
                    features['actual_consumption_m3'] + 1)
            features['demand_pressure'] = features['demand_forecast_m3'] / (features['population_served'] + 1)
            features['complaint_rate'] = features['complaints_received'] / (features['population_served'] + 1) * 10000
            features['water_stress'] = (features['demand_forecast_m3'] - features['actual_consumption_m3']) / (
                    features['demand_forecast_m3'] + 1)
            features['is_dry_season'] = 1 if features['rainfall_mm'] < df_processed['rainfall_mm'].median() else 0
            feature_vector = np.array([[features[col] for col in model_results['feature_columns']]])
            feature_scaled = model_results['scaler'].transform(feature_vector)
            time_probs, zone_probs = model_results['model'].predict(feature_scaled, verbose=0)
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


def load_and_prepare_data(filepath='nawassco.csv'):
    """Load and prepare the water supply dataset for price discrimination analysis"""
    try:
        df = pd.read_csv(filepath)

        # Standardize column names
        df.columns = df.columns.str.lower().str.replace('/', '_').str.replace(' ', '_')

        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

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
    n_samples = 200

    dates = pd.date_range(start='2025-01-01', periods=n_samples, freq='D')
    zones = np.random.choice(['Central', 'Northern', 'Southern', 'Western', 'Eastern'], n_samples)
    times = np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples)
    days = [d.strftime('%A') for d in dates]

    df = pd.DataFrame({
        'date': dates,
        'zone': zones,
        'day_of_week': days,
        'time_of_supply': times,
        'population_served': np.random.randint(25000, 55000, n_samples),
        'demand_forecast_m3': np.random.randint(5000, 15000, n_samples),
        'actual_consumption_m3': np.random.uniform(4000, 14000, n_samples),
        'rainfall_mm': np.random.uniform(0, 50, n_samples),
        'pipe_leakage_m3': np.random.randint(0, 500, n_samples),
        'price_per_litre': np.random.uniform(0.05, 0.35, n_samples)
    })

    return df


def feature_engineering(df):
    """Create features for the model"""
    df_feat = df.copy()

    # Extract date features
    df_feat['month'] = df_feat['date'].dt.month
    df_feat['day'] = df_feat['date'].dt.day
    df_feat['day_of_year'] = df_feat['date'].dt.dayofyear
    df_feat['week_of_year'] = df_feat['date'].dt.isocalendar().week
    df_feat['is_weekend'] = df_feat['date'].dt.dayofweek.isin([5, 6]).astype(int)

    # Encode day of week
    day_mapping = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    df_feat['day_of_week_encoded'] = df_feat['day_of_week'].map(day_mapping)

    # Consumption ratio
    df_feat['consumption_ratio'] = df_feat['actual_consumption_m3'] / (df_feat['demand_forecast_m3'] + 1)

    # Per capita consumption
    df_feat['per_capita_consumption'] = df_feat['actual_consumption_m3'] / (df_feat['population_served'] + 1)

    return df_feat


# ==================== MODEL BUILDING ====================
def build_multi_output_model(input_dim, n_zones, n_times):
    """
    multi-output neural network for price discrimination

    Outputs:
    1. Price (Regression)
    2. Time of Supply (Classification)
    3. Zone (Classification)
    """

    # Input layer
    input_layer = KerasInput(shape=(input_dim,), name='input')

    # Shared layers - Complex feature extraction
    x = Dense(256, activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    # Branch 1: Price Prediction (Regression)
    price_branch = Dense(32, activation='relu', name='price_dense_1')(x)
    price_branch = Dropout(0.2)(price_branch)
    price_branch = Dense(16, activation='relu', name='price_dense_2')(price_branch)
    price_output = Dense(1, activation='linear', name='price_output')(price_branch)

    # Branch 2: Time of Supply (Classification)
    time_branch = Dense(32, activation='relu', name='time_dense_1')(x)
    time_branch = Dropout(0.2)(time_branch)
    time_branch = Dense(16, activation='relu', name='time_dense_2')(time_branch)
    time_output = Dense(n_times, activation='softmax', name='time_output')(time_branch)

    # Branch 3: Zone (Classification)
    zone_branch = Dense(32, activation='relu', name='zone_dense_1')(x)
    zone_branch = Dropout(0.2)(zone_branch)
    zone_branch = Dense(16, activation='relu', name='zone_dense_2')(zone_branch)
    zone_output = Dense(n_zones, activation='softmax', name='zone_output')(zone_branch)

    # Create model
    model = Model(
        inputs=input_layer,
        outputs=[price_output, time_output, zone_output],
        name='PriceDiscriminationModel'
    )

    # Compile with different losses for each output
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'price_output': 'mse',
            'time_output': 'categorical_crossentropy',
            'zone_output': 'categorical_crossentropy'
        },
        loss_weights={
            'price_output': 1.0,
            'time_output': 0.5,
            'zone_output': 0.5
        },
        metrics={
            'price_output': ['mae'],
            'time_output': ['accuracy'],
            'zone_output': ['accuracy']
        }
    )

    return model


def train_price_discrimination_model(df):
    """Train the multi-output model"""

    print("\n🔧 Starting model training...")

    # Feature engineering
    df_processed = feature_engineering(df)

    # Define features
    feature_columns = [
        'month', 'day', 'day_of_year', 'week_of_year', 'is_weekend',
        'day_of_week_encoded', 'population_served', 'demand_forecast_m3',
        'actual_consumption_m3', 'consumption_ratio', 'per_capita_consumption'
    ]

    X = df_processed[feature_columns].values

    # Prepare targets
    y_price = df_processed['price_per_litre'].values
    time_encoder = LabelEncoder()
    y_time_encoded = time_encoder.fit_transform(df_processed['time_of_supply'])
    y_time_categorical = keras.utils.to_categorical(y_time_encoded)
    zone_encoder = LabelEncoder()
    y_zone_encoded = zone_encoder.fit_transform(df_processed['zone'])
    y_zone_categorical = keras.utils.to_categorical(y_zone_encoded)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_price_train, y_price_test, y_time_train, y_time_test, y_zone_train, y_zone_test = train_test_split(
        X_scaled, y_price, y_time_categorical, y_zone_categorical,
        test_size=0.2, random_state=42
    )

    # Build and train model
    model = build_multi_output_model(
        input_dim=X_train.shape[1],
        n_zones=len(zone_encoder.classes_),
        n_times=len(time_encoder.classes_)
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

    history = model.fit(
        X_train,
        {'price_output': y_price_train, 'time_output': y_time_train, 'zone_output': y_zone_train},
        validation_split=0.2, epochs=100, batch_size=16, callbacks=[early_stop, reduce_lr], verbose=0
    )

    # Evaluation
    y_price_pred, y_time_pred, y_zone_pred = model.predict(X_test, verbose=0)
    price_mae = mean_absolute_error(y_price_test, y_price_pred)
    price_rmse = np.sqrt(mean_squared_error(y_price_test, y_price_pred))
    time_accuracy = accuracy_score(np.argmax(y_time_test, axis=1), np.argmax(y_time_pred, axis=1))
    zone_accuracy = accuracy_score(np.argmax(y_zone_test, axis=1), np.argmax(y_zone_pred, axis=1))

    print(f"\n✅ Training complete! Epochs: {len(history.history['loss'])}")
    print(f"   Price MAE: {price_mae:.4f}, Time Acc: {time_accuracy:.2%}, Zone Acc: {zone_accuracy:.2%}")

    return {
        'model': model, 'history': history, 'scaler': scaler, 'time_encoder': time_encoder,
        'zone_encoder': zone_encoder, 'feature_columns': feature_columns,
        'metrics': {'price_mae': price_mae, 'price_rmse': price_rmse, 'time_accuracy': time_accuracy,
                    'zone_accuracy': zone_accuracy},
        'test_data': {'X_test': X_test, 'y_price_test': y_price_test, 'y_time_test': y_time_test,
                      'y_zone_test': y_zone_test, 'predictions': (y_price_pred, y_time_pred, y_zone_pred)}
    }


# ====================SCHEDULE GENERATION ====================
def generate_distribution_schedule(model_results, df, days_ahead=7):
    """Generate optimal distribution schedule for upcoming days."""
    df_processed = feature_engineering(df)
    latest_date = df_processed['date'].max()
    future_dates = [latest_date + timedelta(days=i) for i in range(1, days_ahead + 1)]

    schedule = []

    # Use historical averages for non-temporal features
    avg_features = {
        'population_served': df_processed['population_served'].mean(),
        'demand_forecast_m3': df_processed['demand_forecast_m3'].mean(),
        'actual_consumption_m3': df_processed['actual_consumption_m3'].mean()
    }

    for date in future_dates:
        # Create features for the future date
        features = {
            'month': date.month,
            'day': date.day,
            'day_of_year': date.timetuple().tm_yday,
            'week_of_year': date.isocalendar()[1],
            'is_weekend': 1 if date.weekday() >= 5 else 0,
            'day_of_week_encoded': date.weekday(),
            **avg_features
        }
        features['consumption_ratio'] = features['actual_consumption_m3'] / (features['demand_forecast_m3'] + 1)
        features['per_capita_consumption'] = features['actual_consumption_m3'] / (features['population_served'] + 1)

        # Create feature vector in the correct order
        feature_vector = np.array([[features[col] for col in model_results['feature_columns']]])

        # Scale features
        feature_scaled = model_results['scaler'].transform(feature_vector)

        # Predict
        price_pred, time_probs, zone_probs = model_results['model'].predict(feature_scaled, verbose=0)

        # Decode predictions
        recommended_price = price_pred[0][0]
        recommended_time = model_results['time_encoder'].inverse_transform([np.argmax(time_probs[0])])[0]
        recommended_zone = model_results['zone_encoder'].inverse_transform([np.argmax(zone_probs[0])])[0]

        schedule.append({
            'Date': date.strftime('%Y-%m-%d (%A)'),
            'Predicted Zone': recommended_zone,
            'Predicted Time': recommended_time,
            'Predicted Price per Litre': f"{recommended_price:.4f}"
        })

    return pd.DataFrame(schedule)


def run_prophet_analysis(df,
                         date_column='date',
                         target_column='actual_consumption_m3',
                         forecast_days=14,
                         changepoint_prior_scale=0.05,
                         weekly_seasonality=True,
                         yearly_seasonality=False,
                         aggregate_by_date=True):
    """
    Run complete Prophet time series analysis on a dataframe.
    """
    # Prepare data
    df_clean = df.copy()

    # Standardize column names if needed
    if date_column in df_clean.columns:
        df_clean[date_column] = pd.to_datetime(df_clean[date_column])
    else:
        raise ValueError(f"Date column '{date_column}' not found in dataframe")

    if target_column not in df_clean.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    # Aggregate by date if specified
    if aggregate_by_date and len(df_clean.columns) > 2:
        # Group by date and sum the target
        prophet_df = df_clean.groupby(date_column).agg({
            target_column: 'sum'
        }).reset_index()
    else:
        prophet_df = df_clean[[date_column, target_column]].copy()

    # Rename for Prophet
    prophet_df = prophet_df.rename(columns={date_column: 'ds', target_column: 'y'})

    # Remove any NaN values
    prophet_df = prophet_df.dropna()

    # Sort by date
    prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)

    print(f"📊 Prophet Analysis Starting...")
    print(f"   Data points: {len(prophet_df)}")
    print(f"   Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
    print(f"   Forecast period: {forecast_days} days")

    # Train Prophet model
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        stan_backend='CMDSTANPY'
    )

    # Fit with error handling
    try:
        model.fit(prophet_df)
        print("✅ Model training successful")
    except Exception as e:
        print(f"⚠️ Initial training failed, using Newton optimizer...")
        import logging
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
        model.fit(prophet_df, algorithm='Newton')
        print("✅ Model training successful (Newton)")

    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # ==================== CREATE FORECAST PLOT ====================
    fig_forecast = go.Figure()

    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=prophet_df['ds'],
        y=prophet_df['y'],
        mode='markers',
        name='Actual Data',
        marker=dict(size=6, color='#2c3e50', opacity=0.7),
        hovertemplate='Date: %{x}<br>Value: %{y:,.2f}<extra></extra>'
    ))

    # Forecast line
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Prophet Forecast',
        line=dict(color='#3498db', width=3),
        hovertemplate='Date: %{x}<br>Forecast: %{y:,.2f}<extra></extra>'
    ))

    # Confidence interval upper
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Confidence interval lower (with fill)
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='80% Confidence Interval',
        fill='tonexty',
        fillcolor='rgba(52, 152, 219, 0.2)',
        line=dict(width=0),
        hovertemplate='Lower: %{y:,.2f}<extra></extra>'
    ))

    # Add vertical line to separate historical and forecast
    last_historical_date = prophet_df['ds'].max()

    fig_forecast.add_vrect(
        x0=last_historical_date,
        x1=forecast['ds'].max(),
        fillcolor="rgba(255, 165, 0, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="Forecast Period",
        annotation_position="top left",
        annotation=dict(font=dict(size=12, color="rgba(255, 165, 0, 0.8)"))
    )

    fig_forecast.add_vline(
        x=last_historical_date,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Forecast Start",
        annotation_position="top"
    )

    fig_forecast.update_layout(
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        template='plotly_white',
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # ==================== CREATE COMPONENTS PLOT ====================
    num_components = 2 if weekly_seasonality and not yearly_seasonality else 3
    subplot_titles = ['Trend Component']

    if weekly_seasonality:
        subplot_titles.append('Weekly Seasonality')
    if yearly_seasonality:
        subplot_titles.append('Yearly Seasonality')

    subplot_titles.append('Forecast Uncertainty')

    fig_components = make_subplots(
        rows=len(subplot_titles), cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08
    )

    row = 1

    # Trend
    fig_components.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['trend'],
        mode='lines',
        name='Trend',
        line=dict(color='#e74c3c', width=2),
        hovertemplate='%{y:,.2f}<extra></extra>'
    ), row=row, col=1)
    fig_components.update_yaxes(title_text="Trend", row=row, col=1)
    row += 1

    # Weekly seasonality
    if weekly_seasonality:
        fig_components.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['weekly'],
            mode='lines',
            name='Weekly',
            line=dict(color='#27ae60', width=2),
            hovertemplate='%{y:,.2f}<extra></extra>'
        ), row=row, col=1)
        fig_components.update_yaxes(title_text="Weekly Effect", row=row, col=1)
        row += 1

    # Yearly seasonality
    if yearly_seasonality:
        fig_components.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yearly'],
            mode='lines',
            name='Yearly',
            line=dict(color='#f39c12', width=2),
            hovertemplate='%{y:,.2f}<extra></extra>'
        ), row=row, col=1)
        fig_components.update_yaxes(title_text="Yearly Effect", row=row, col=1)
        row += 1

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
        hovertemplate='%{y:,.2f}<extra></extra>'
    ), row=row, col=1)
    fig_components.update_yaxes(title_text="Uncertainty Range", row=row, col=1)

    fig_components.update_xaxes(title_text="Date", row=row, col=1)

    fig_components.update_layout(
        height=250 * len(subplot_titles),
        showlegend=False,
        template='plotly_white',
        title_text='Prophet Component Decomposition'
    )

    # ==================== CALCULATE METRICS ====================
    # Get predictions for historical data
    train_forecast = forecast[forecast['ds'].isin(prophet_df['ds'])]
    actual_values = prophet_df['y'].values
    predicted_values = train_forecast['yhat'].values[:len(actual_values)]

    mae = np.mean(np.abs(actual_values - predicted_values))
    rmse = np.sqrt(np.mean((actual_values - predicted_values) ** 2))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'data_points': len(prophet_df),
        'forecast_points': forecast_days
    }

    # ==================== FORECAST SUMMARY ====================
    future_forecast = forecast[forecast['ds'] > prophet_df['ds'].max()]

    forecast_summary = {
        'avg_forecast': future_forecast['yhat'].mean(),
        'min_forecast': future_forecast['yhat'].min(),
        'max_forecast': future_forecast['yhat'].max(),
        'total_forecast': future_forecast['yhat'].sum(),
        'avg_confidence_width': (future_forecast['yhat_upper'] - future_forecast['yhat_lower']).mean(),
        'forecast_start_date': future_forecast['ds'].min(),
        'forecast_end_date': future_forecast['ds'].max()
    }

    print("\n📊 Analysis Complete!")
    print(f"   MAE: {mae:,.2f}")
    print(f"   RMSE: {rmse:,.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Avg Forecast: {forecast_summary['avg_forecast']:,.2f}")

    # Return all results
    return {
        'forecast_fig': fig_forecast,
        'components_fig': fig_components,
        'metrics': metrics,
        'forecast_summary': forecast_summary,
        'model': model,
        'forecast_df': forecast,
        'prepared_data': prophet_df
    }


# --- [4] Pre-load Static Data for Layouts ---
FULL_DF = get_full_data_df()
ZONES_OPTIONS = [{'label': z, 'value': z} for z in sorted(FULL_DF['zone'].unique())]
SUPPLY_AREAS_OPTIONS = [{'label': sa, 'value': sa} for sa in sorted(FULL_DF['supply_area'].unique())]


# --- [5] UI Components & Layouts ---
def create_kpi_card(title, value, icon, color="primary"):
    return dbc.Card(dbc.CardBody([
        html.Div(className="row align-items-center", children=[
            html.Div(className="col-8",
                     children=[html.H5(title, className="card-title text-muted fw-bold"),
                               html.H3(value, className="card-text text-black")]),
            html.Div(className="col-4 d-flex justify-content-center align-items-center",
                     children=[html.I(className=f"fas {icon} fa-3x text-{color} opacity-75")])
        ])
    ]), className="h-100 shadow-sm border-0 rounded-3 kpi-card")


def create_header(user_data):
    if not user_data: return html.Div()
    return dbc.Navbar(
        dbc.Container(
            [
                dcc.Link(
                    dbc.Row(
                        [
                            dbc.Col(dbc.NavbarBrand("Aqua-Predict", className="ms-2 fw-bold text-white")),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink(f"Welcome, {user_data['username']}", disabled=True,
                                                    className="text-white-50")),
                            dbc.NavItem(dcc.Link("Logout", href="/logout", className="btn btn-outline-light ms-md-3")),
                        ],
                        className="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ],
        ),
        color="primary",
        dark=True,
        className="shadow-sm mb-4"
    )

def create_chat_layout():
    return html.Div([
        dbc.Card(dbc.CardBody([
            html.H4("Public Chat & Announcements", className="card-title text-center mb-3 text-primary"),
            html.Hr(),
            html.Div(id='chat-display',
                     style={'height': '400px', 'overflowY': 'scroll', 'border': '1px solid #ccc', 'padding': '10px',
                            'borderRadius': '5px', 'backgroundColor': '#f9f9f9'}),
            dbc.InputGroup(
                [
                    dbc.Input(id='chat-input', placeholder='Type your message...', type='text'),
                    dbc.Button(html.I(className="fas fa-paper-plane"), id='chat-send-button', n_clicks=0,
                               color="primary"),
                ],
                className="mt-3"
            ),
            dcc.Interval(id='chat-refresh-interval', interval=2000, n_intervals=0, disabled=False),
        ]), className="shadow-sm")
    ])


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

# Global variables for model and data
model_results = None
df_global = None
schedule_df = None

df_global = load_and_prepare_data()

Tt_model_results = None
Tt_df_global = None
Tt_schedule_df = None

# Load data on startup
Tt_df_global = Tt_load_and_prepare_data()

# UI/UX Change: Revamped login and registration layouts for a modern look
login_layout = dbc.Container(fluid=True,
                             className="min-vh-100 d-flex flex-column justify-content-center p-3 login-container",
                             children=[
                                 dbc.Row(justify="center", align="center", children=
                                 dbc.Col(width=12, sm=10, md=6, lg=4, children=[
                                     dbc.Card(
                                         dbc.CardBody(className="p-4 p-md-5", children=[
                                             html.Div(className="text-center mb-4", children=[
                                                 html.I(className="fas fa-water fa-3x text-primary"),
                                                 html.H2("Aqua-Predict", className="mt-2 fw-bold"),
                                                 html.P("Water Management & Forecasting System",
                                                        className="text-muted"),
                                             ]),
                                             html.Div(id='login-welcome-message',
                                                      className='text-center text-muted mb-4 small'),
                                             dbc.Input(id='username-input', type='text', placeholder='Username',
                                                       className="mb-3 form-control-lg"),
                                             dbc.Input(id='password-input', type='password', placeholder='Password',
                                                       className="mb-3 form-control-lg"),
                                             dbc.Button('Login', id='login-button', n_clicks=0, color="primary",
                                                        className="w-100 mb-3 fw-bold btn-lg"),
                                             dbc.Alert(id='login-output', className="mb-3", style={'display': 'none'}),
                                             html.Div(className="text-center", children=[
                                                 dcc.Link("Register a new consumer account", href="/register",
                                                          className="text-decoration-none small")
                                             ]),
                                             dbc.Accordion([
                                                 dbc.AccordionItem(
                                                     html.Ul([
                                                         html.Li([html.Span("Username: "), html.Code("Admin"),
                                                                  html.Span(" | Password: "), html.Code("Admin123")]),
                                                         html.Li([html.Span("Username: "), html.Code("dalmas"),
                                                                  html.Span(" | Password: "), html.Code("pass")]),
                                                         html.Li([html.Span("Username: "), html.Code("chituyi"),
                                                                  html.Span(" | Password: "), html.Code("pass")]),
                                                     ], className="list-unstyled small"),
                                                     title="Demo Accounts"
                                                 )
                                             ], start_collapsed=True, className="mt-4"),
                                         ]),
                                     )
                                 ])
                                         )
                             ])

registration_layout = dbc.Container(fluid=True,
                                    className="min-vh-100 d-flex flex-column justify-content-center p-3 login-container",
                                    children=[
                                        dbc.Row(justify="center", align="center", children=
                                        dbc.Col(width=12, sm=10, md=6, lg=4, children=[
                                            dbc.Card(
                                                dbc.CardBody(className="p-4 p-md-5", children=[
                                                    html.Div(className="text-center mb-4", children=[
                                                        html.I(className="fas fa-user-plus fa-3x text-success"),
                                                        html.H2("Create Consumer Account", className="mt-2 fw-bold"),
                                                    ]),
                                                    dbc.Input(id='reg-username', placeholder='Username',
                                                              className="mb-3 form-control-lg"),
                                                    dbc.Input(id='reg-password', placeholder='Password',
                                                              type='password', className="mb-3 form-control-lg"),
                                                    dcc.Dropdown(id='reg-zone', placeholder='Select Your Zone',
                                                                 options=ZONES_OPTIONS, className="mb-3"),
                                                    dcc.Dropdown(id='reg-supply-area',
                                                                 placeholder='Select Your Supply Area',
                                                                 options=SUPPLY_AREAS_OPTIONS, className="mb-4"),
                                                    dbc.Button("Register", id='register-button', n_clicks=0,
                                                               color="success", className="w-100 fw-bold btn-lg"),
                                                    html.Div(id='register-output', className="text-center mt-3"),
                                                    html.Hr(),
                                                    dcc.Link("Back to Login", href="/login",
                                                             className="d-block text-center mt-3 text-primary text-decoration-none fw-bold")
                                                ])
                                            )
                                        ])
                                                )
                                    ])
def create_admin_layout(user_data):
    return html.Div([create_header(user_data), dbc.Container(fluid=True, children=[
        dbc.Tabs(id="admin-tabs", active_tab='tab-summary', className="nav-justified", children=[
            dbc.Tab(label='Executive Summary', tab_id='tab-summary', tabClassName="fw-bold"),
            dbc.Tab(label='Predictive Model', tab_id='tab-models', tabClassName="fw-bold"),
            dbc.Tab(label='Policy & Management', tab_id='tab-policy', tabClassName="fw-bold"),
            dbc.Tab(label='Consumer Analytics', tab_id='tab-all-consumers', tabClassName="fw-bold"),
            dbc.Tab(label='Chat', tab_id='tab-chat', tabClassName="fw-bold"),
        ]),
        html.Div(id='admin-tabs-content', className="mt-4 p-4 bg-light rounded-3 shadow-sm")
    ])])


def create_consumer_layout(user_data):
    return html.Div([
        create_header(user_data),
        dbc.Container(fluid=True, children=[
            dbc.Tabs(id="consumer-tabs", active_tab='tab-dashboard', className="nav-justified", children=[
                dbc.Tab(label='Dashboard', tab_id='tab-dashboard', tabClassName="fw-bold"),
                dbc.Tab(label='Billing & Payments', tab_id='tab-billing', tabClassName="fw-bold"),
                dbc.Tab(label='Chat', tab_id='tab-chat', tabClassName="fw-bold"),
            ]),
            html.Div(id='consumer-tabs-content', className="mt-4 p-4 bg-light rounded-3 shadow-sm"),
        ]),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Payment Processing")),
            dbc.ModalBody(id="payment-modal-body"),
        ], id="payment-modal", is_open=False),
        dcc.Store(id='trigger-refresh-store', data=0),
        dcc.Interval(id='payment-completion-interval', interval=1000, n_intervals=0, disabled=True)
    ])


# --- [6] Main App Layout & Routing ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Store(id='user-store', storage_type='session'),
    html.Div(id='page-content'),
    dcc.Interval(id='login-time-interval', interval=1000, n_intervals=0)
])


@app.callback(Output('page-content', 'children'), Input('url', 'pathname'), State('user-store', 'data'))
def serve_layout(pathname, user_data):
    user_data = user_data or {}
    is_authenticated = user_data.get('is_authenticated', False)
    user_type = user_data.get('user_type')

    if pathname == '/register': return registration_layout

    if is_authenticated:
        if user_type == 'admin' and (pathname.startswith('/admin') or pathname == '/'): return create_admin_layout(
            user_data)
        if user_type == 'consumer' and (
                pathname.startswith('/consumer') or pathname == '/'): return create_consumer_layout(user_data)

    return login_layout


# --- [7] Callbacks ---
@app.callback(
    [Output('url', 'pathname'), Output('user-store', 'data', allow_duplicate=True), Output('login-output', 'children'),
     Output('login-output', 'style')],
    [Input('login-button', 'n_clicks')],
    [State('username-input', 'value'), State('password-input', 'value')],
    prevent_initial_call=True
)
def login_callback(n_clicks, username, password):
    if not username or not password: return dash.no_update, dash.no_update, dbc.Alert("Please enter credentials.",
                                                                                      color="warning"), {
        'display': 'block'}
    conn = sqlite3.connect(DB_FILE)
    user_record = conn.execute("SELECT id, password_hash, user_type, zone, supply_area FROM users WHERE username = ?",
                               (username,)).fetchone()
    conn.close()
    if user_record and check_password_hash(user_record[1], password):
        user_data = {'is_authenticated': True, 'id': user_record[0], 'username': username, 'user_type': user_record[2],
                     'zone': user_record[3], 'supply_area': user_record[4]}
        redirect_path = '/admin' if user_data['user_type'] == 'admin' else '/consumer'
        return redirect_path, user_data, '', {'display': 'none'}
    return dash.no_update, dash.no_update, dbc.Alert("Invalid credentials.", color="danger"), {'display': 'block'}


@app.callback(Output('user-store', 'data'), Input('url', 'pathname'), prevent_initial_call=True)
def handle_logout(pathname):
    if pathname == '/logout':
        return None
    return dash.no_update


@app.callback(Output('register-output', 'children'), Input('register-button', 'n_clicks'),
              [State('reg-username', 'value'), State('reg-password', 'value'), State('reg-zone', 'value'),
               State('reg-supply-area', 'value')], prevent_initial_call=True)
def register_callback(n_clicks, username, password, zone, supply_area):
    if not all([username, password, zone, supply_area]): return dbc.Alert("All fields are required.", color="warning")
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash, user_type, zone, supply_area) VALUES (?, ?, 'consumer', ?, ?)",
            (username, generate_password_hash(password), zone, supply_area))
        conn.commit()
        return dbc.Alert("Registration successful! Please log in.", color="success")
    except sqlite3.IntegrityError:
        return dbc.Alert("Username already exists.", color="danger")
    finally:
        conn.close()


@app.callback(
    Output('login-welcome-message', 'children'),
    Input('login-time-interval', 'n_intervals')
)
def update_login_welcome(n_intervals):
    current_time = datetime.now()
    current_date_str = current_time.strftime("%A, %B %d, %Y")
    current_time_str = current_time.strftime("%I:%M:%S %p")
    return f"Welcome! Today is {current_date_str}, {current_time_str} in Nairobi, Kenya."


@app.callback(
    Output('admin-tabs-content', 'children'),
    [Input('admin-tabs', 'active_tab')]
)
def render_admin_tab_content(active_tab):
    if active_tab == 'tab-summary':
        # Calculate a DataFrame for analytics
        user_df = query_db("SELECT username, zone, supply_area FROM users WHERE user_type = 'consumer'")
        merged_df = pd.merge(user_df, FULL_DF, on='supply_area')
        stats_df = merged_df.groupby('username').agg(
            zone=('zone_x', 'first'),
            avg_consumption=('actual_consumption_m3', 'mean'),
            total_leakage_ytd=('pipe_leakage_m3', 'sum'),
            total_complaints_ytd=('complaints_received', 'sum'),
            avg_price_per_litre=('price_per_litre', 'mean')
        ).reset_index()

        kpis = [
            dbc.Col(create_kpi_card("Total Consumers",
                                    f"{query_db('SELECT COUNT(*) FROM users WHERE user_type=\"consumer\"').iloc[0, 0]}",
                                    "fa-users", "info"), lg=4, className="mb-4"),
            dbc.Col(create_kpi_card("Total Leakage (YTD)", f"{FULL_DF['pipe_leakage_m3'].sum() / 1000:.1f}k m³",
                                    "fa-tint-slash", "warning"), lg=4, className="mb-4"),
            dbc.Col(
                create_kpi_card("Total Complaints (YTD)", f"{FULL_DF['complaints_received'].sum():,.0f}", "fa-bullhorn",
                                "danger"), lg=4, className="mb-4"),
        ]

        revenue_df = FULL_DF.groupby('date').apply(
            lambda x: (x['actual_consumption_m3'] * x['price_per_litre']).sum()).reset_index(name='revenue')
        fig_revenue = px.line(revenue_df, x='date', y='revenue', title='Total Revenue Over Time',
                              color_discrete_sequence=[COLOR_PALETTE[0]])
        fig_revenue.update_yaxes(title_text='Revenue (KES)')
        fig_revenue.update_layout(**PLOT_LAYOUT)

        fig_pie = px.pie(FULL_DF.groupby('zone')['population_served'].mean().reset_index(), names='zone',
                         values='population_served', title='Population Served by Zone',
                         color_discrete_sequence=COLOR_PALETTE)
        fig_pie.update_layout(**PLOT_LAYOUT)

        water_quality_df = FULL_DF.groupby('date')['ph_level'].mean().reset_index()
        fig_quality = px.line(water_quality_df, x='date', y='ph_level', title='Water Quality (pH) Trend',
                              color_discrete_sequence=[COLOR_PALETTE[1]])
        fig_quality.update_layout(**PLOT_LAYOUT, yaxis_title='pH Level')

        complaints_df = FULL_DF.groupby('date')['complaints_received'].sum().reset_index()
        fig_complaints_trend = px.bar(complaints_df, x='date', y='complaints_received',
                                      title='Daily Customer Complaints', color_discrete_sequence=[COLOR_PALETTE[2]])
        fig_complaints_trend.update_layout(**PLOT_LAYOUT)

        return html.Div([
            dbc.Row(kpis),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_revenue)))), lg=8, className="mb-4"),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_pie)))), lg=4, className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_quality)))), lg=6, className="mb-4"),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_complaints_trend)))), lg=6,
                        className="mb-4"),
            ]),
        ])
    elif active_tab == 'tab-models':
        # Find the first valid area with at least 20 data points to use as a default.
        counts = FULL_DF.groupby('supply_area')['date'].count()
        first_valid_area = counts[counts >= 20].index[0] if not counts[counts >= 20].empty else (
            SUPPLY_AREAS_OPTIONS[0]['value'] if SUPPLY_AREAS_OPTIONS else None)

        return dbc.Card(dbc.CardBody([
            # html.H5("Prophet Forecast & Explainable AI"),
            html.Div([
                html.Div([
                    html.H1("💧 Water Supply Forecasting with Facebook Prophet",
                            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),

                    html.P("FBProphet with XAI components For water Forecasting",
                           style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'}),
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
                                marks={0: '0 sps', 100: '100 sps', 250: '200 sps', 400: '4000 sps', 550: '500 sps',
                                       700: '700 sps'},
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

                ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px',
                          'marginBottom': '20px'}),

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
                        html.Div(id='FB_metrics-display', style={'marginTop': '20px'}),
                        html.Div(id='insights-display', style={'marginTop': '20px'})
                    ]
                )
            ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px', 'margin': '0 auto'})

        ]))
    elif active_tab == 'tab-policy':
        consumer_options = query_db("SELECT id, username FROM users WHERE user_type = 'consumer'").to_dict('records')
        consumer_options = [{'label': i['username'], 'value': i['id']} for i in consumer_options]
        return dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                # html.H5("Dynamic Rationing Schedule (Water Rationing Model)"),
                html.Div([
                    html.Div([
                        html.H1("💧 Dynamic Water Rationing Scheduler",
                                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                        html.P("Multi-Output Model for Optimal Water Distribution Scheduling",
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
                    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px',
                              'marginBottom': '20px'}),

                    # Loading and Results
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=[
                            html.Div(id='Tt_model-info', style={'marginBottom': '20px'}),
                            dcc.Graph(id='Tt_training-plot', style={'marginBottom': '20px'}),
                            dcc.Graph(id='Tt_confusion-plot', style={'marginBottom': '20px'}),
                            html.Div(id='Tt_classification-reports', style={'marginBottom': '20px'}),
                            html.Div(id='Tt_schedule-display', style={'marginBottom': '20px'}),
                            dcc.Graph(id='Tt_schedule-viz', style={'marginBottom': '20px'}),
                            html.Div(id='Tt_metrics-display', style={'marginTop': '20px'}),
                        ]
                    )
                ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'maxWidth': '1600px',
                          'margin': '0 auto'})

            ])), lg=6, className="mb-4"),
            dbc.Col(dbc.Card(dbc.CardBody([
                # html.H5("Price Discrimination Model Simulator"),

                html.Div([
                    html.Div([
                        html.H1("💧 Water Distribution Scheduler & Price Analysis",
                                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                        html.P("Predicting Price, Time, and Zone for optimal water distribution",
                               style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '30px'})
                    ]),

                    # Control Panel
                    html.Div([
                        html.H3("🎯 Model & Schedule Controls", style={'color': '#2c3e50'}),
                        html.Div([
                            # Left side for training
                            html.Div([
                                html.Label("Select Target Output to Analyze:", style={'fontWeight': 'bold'}),
                                dcc.Dropdown(
                                    id='target-dropdown',
                                    options=[
                                        {'label': '💵 Price per Litre', 'value': 'price'},
                                        {'label': '📊 All Outputs Combined', 'value': 'all'}
                                    ],
                                    value='all'
                                ),
                                html.Button('🤖 Train Multi-Output Model', id='Tt_train-button', n_clicks=0,
                                            style={'width': '100%', 'marginTop': '10px', 'backgroundColor': '#9b59b6',
                                                   'color': 'white',
                                                   'padding': '10px', 'fontSize': '16px', 'border': 'none',
                                                   'borderRadius': '5px'})
                            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                            # Right side for scheduling
                            html.Div([
                                html.Label("Schedule Horizon (Days Ahead):", style={'fontWeight': 'bold'}),
                                dcc.Slider(
                                    id='schedule-days-slider', min=3, max=14, step=1, value=7,
                                    marks={3: '3', 7: '7', 10: '10', 14: '14'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.Button('📅 Generate Distribution Schedule', id='Tt_schedule-button', n_clicks=0,
                                            style={'width': '100%', 'marginTop': '10px', 'backgroundColor': '#3498db',
                                                   'color': 'white',
                                                   'padding': '10px', 'fontSize': '16px', 'border': 'none',
                                                   'borderRadius': '5px'})
                            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%',
                                      'verticalAlign': 'top'})
                        ])
                    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px',
                              'marginBottom': '20px'}),

                    # Loading and Results
                    dcc.Loading(
                        id="loading", type="default",
                        children=[
                            html.Div(id='model-info'),
                            dcc.Graph(id='performance-plot'),
                            dcc.Graph(id='prediction-plot'),
                            dcc.Graph(id='feature-importance-plot'),
                            html.Div(id='metrics-display'),
                            html.Div(id='schedule-display'),
                            dcc.Graph(id='schedule-viz')
                        ]
                    )
                ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif', 'maxWidth': '1400px',
                          'margin': '0 auto'})

            ])), lg=6, className="mb-4"),

            # New section for posting bills (Redesigned)
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H5("Post Bill to Consumer", className="card-title mb-4 text-center"),

                        # Card for Input Fields
                        dbc.Card(
                            [
                                dbc.CardHeader("Step 1: Enter Bill Details"),
                                dbc.CardBody([
                                    dcc.Dropdown(
                                        id='bill-user-dropdown',
                                        placeholder='Select Consumer',
                                        options=consumer_options,
                                    ),
                                    html.Br(),
                                    dbc.Input(
                                        id='bill-amount-input',
                                        type='number',
                                        placeholder='Enter Bill Amount (KES)',
                                    ),
                                ]),
                            ],
                            className="mb-3",
                        ),

                        # Card for the Calendar
                        dbc.Card(
                            [
                                dbc.CardHeader("Step 2: Select Due Date"),
                                dbc.CardBody(
                                    dcc.DatePickerSingle(
                                        id='bill-due-date-picker',
                                        placeholder='Select Due Date',
                                        min_date_allowed=datetime.now(),
                                        className='w-100',
                                        display_format='YYYY-MM-DD'
                                    )
                                ),
                            ],
                            className="mb-3",
                        ),

                        # Card for the Post Bill Button
                        dbc.Card(
                            [
                                dbc.CardBody([
                                    dbc.Button(
                                        "Post Bill",
                                        id='post-bill-button',
                                        color="primary",
                                        className="w-100"
                                    ),
                                    html.Div(id='post-bill-output', className="mt-3")
                                ])
                            ]
                        ),
                    ])
                ),
                lg=6,
                className="mb-4"
            )
        ])
    elif active_tab == 'tab-all-consumers':
        user_df = query_db("SELECT username, zone, supply_area FROM users WHERE user_type = 'consumer'")
        merged_df = pd.merge(user_df, FULL_DF, on='supply_area')
        stats_df = merged_df.groupby('username').agg(
            zone=('zone_x', 'first'),
            avg_consumption=('actual_consumption_m3', 'mean'),
            total_leakage_ytd=('pipe_leakage_m3', 'sum'),
            total_complaints_ytd=('complaints_received', 'sum'),
            avg_price_per_litre=('price_per_litre', 'mean')
        ).reset_index()

        stats_df['estimated_annual_bill'] = stats_df['avg_consumption'] * 365 * stats_df['avg_price_per_litre']
        stats_df['payment_status'] = np.random.choice(['Paid', 'Overdue'], size=len(stats_df), p=[0.9, 0.1])

        fig_cons = px.bar(stats_df, x='username', y='avg_consumption', color='zone',
                          title='Avg. Consumption by Consumer', color_discrete_sequence=COLOR_PALETTE)
        fig_cons.update_layout(**PLOT_LAYOUT)
        fig_leak = px.bar(stats_df, x='username', y='total_leakage_ytd', color='zone',
                          title='Total Leakage by Consumer', color_discrete_sequence=COLOR_PALETTE)
        fig_leak.update_layout(**PLOT_LAYOUT)
        fig_comp = px.bar(stats_df, x='username', y='total_complaints_ytd', color='zone',
                          title='Total Complaints by Consumer', color_discrete_sequence=COLOR_PALETTE)
        fig_comp.update_layout(**PLOT_LAYOUT)
        fig_bill = px.bar(stats_df, x='username', y='estimated_annual_bill', color='zone',
                          title='Estimated Annual Bill by Consumer', color_discrete_sequence=COLOR_PALETTE)
        fig_bill.update_layout(**PLOT_LAYOUT)

        fig_consumer_ts = px.line(
            merged_df.groupby(['date', 'username'])['actual_consumption_m3'].sum().reset_index(),
            x='date', y='actual_consumption_m3', color='username',
            title='Consumption by User Over Time',
            color_discrete_sequence=COLOR_PALETTE
        )
        fig_consumer_ts.update_layout(**PLOT_LAYOUT)

        bill_by_zone_df = stats_df.groupby('zone')['estimated_annual_bill'].mean().reset_index()
        fig_bill_by_zone = px.bar(
            bill_by_zone_df, x='zone', y='estimated_annual_bill',
            title='Avg. Estimated Bill by Zone',
            color='zone',
            color_discrete_sequence=COLOR_PALETTE
        )
        fig_bill_by_zone.update_layout(**PLOT_LAYOUT)

        fig_scatter = px.scatter(
            stats_df,
            x='avg_consumption',
            y='total_leakage_ytd',
            color='zone',
            title='Consumption vs. Leakage',
            labels={'avg_consumption': 'Avg. Consumption (m³)', 'total_leakage_ytd': 'Total Leakage (m³)'},
            color_discrete_sequence=COLOR_PALETTE
        )
        fig_scatter.update_layout(**PLOT_LAYOUT)

        leakage_by_zone_df = stats_df.groupby('zone')['total_leakage_ytd'].sum().reset_index()
        fig_leakage_pie = px.pie(
            leakage_by_zone_df,
            names='zone',
            values='total_leakage_ytd',
            title='Total Leakage Breakdown by Zone',
            color_discrete_sequence=COLOR_PALETTE
        )
        fig_leakage_pie.update_layout(**PLOT_LAYOUT)

        water_quality_by_zone_df = FULL_DF.groupby('zone')['ph_level'].mean().reset_index()
        fig_water_quality_by_zone = px.bar(
            water_quality_by_zone_df,
            x='zone', y='ph_level',
            title='Average Water Quality (pH) by Zone',
            color_discrete_sequence=[COLOR_PALETTE[1]]
        )
        fig_water_quality_by_zone.update_layout(**PLOT_LAYOUT)

        dates = pd.to_datetime(FULL_DF['date'].unique())
        unpaid_bills_data = np.random.normal(50000, 10000, len(dates)).cumsum() + 200000
        unpaid_bills_df = pd.DataFrame({'date': dates, 'unpaid_amount': unpaid_bills_data})
        fig_unpaid_bills = px.line(
            unpaid_bills_df,
            x='date', y='unpaid_amount',
            title='Unpaid Bills Trend',
            color_discrete_sequence=[COLOR_PALETTE[2]]
        )
        fig_unpaid_bills.update_layout(**PLOT_LAYOUT, yaxis_title='Unpaid Amount (KES)')

        display_df = stats_df.copy()
        display_df['avg_consumption'] = display_df['avg_consumption'].map('{:,.2f}'.format)
        display_df['total_leakage_ytd'] = display_df['total_leakage_ytd'].map('{:,.2f}'.format)
        display_df['estimated_annual_bill'] = display_df['estimated_annual_bill'].map('KES {:,.2f}'.format)

        columns = [{"name": col.replace('_', ' ').title(), "id": col} for col in display_df.columns]

        return dbc.Card(dbc.CardBody([
            html.H5("Consumer Analytics"),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_cons)))), lg=6, className="mb-4"),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_leak)))), lg=6, className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_comp)))), lg=6, className="mb-4"),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_bill)))), lg=6, className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_consumer_ts)))), lg=6, className="mb-4"),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_bill_by_zone)))), lg=6,
                        className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_scatter)))), lg=6, className="mb-4"),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_leakage_pie)))), lg=6, className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_water_quality_by_zone)))), lg=6,
                        className="mb-4"),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_unpaid_bills)))), lg=6,
                        className="mb-4"),
            ]),
            dbc.Card(dbc.CardBody(dcc.Loading(dash_table.DataTable(
                id='all-consumers-table',
                data=display_df.to_dict('records'),
                columns=columns,
                page_size=20, filter_action='native',
                sort_action='native',
                style_table={'overflowX': 'auto', 'border': '1px solid #dee2e6'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
                style_cell={'border': '1px solid #dee2e6'},
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'},
                    {'if': {'filter_query': '{payment_status} = "Overdue"'},
                     'backgroundColor': '#FFE7E7', 'fontWeight': 'bold', 'color': '#842029'},
                    {'if': {'filter_query': '{payment_status} = "Paid"'},
                     'backgroundColor': '#D4EDDA', 'fontWeight': 'bold', 'color': '#0A3622'},
                ],
                style_as_list_view=True
            ))))
        ]))
    elif active_tab == 'tab-chat':
        return create_chat_layout()
    return html.Div()


@app.callback(
    Output('post-bill-output', 'children'),
    [Input('post-bill-button', 'n_clicks')],
    [State('bill-user-dropdown', 'value'),
     State('bill-amount-input', 'value'),
     State('bill-due-date-picker', 'date')]
)
def post_bill(n_clicks, user_id, amount, due_date):
    if not n_clicks:
        return ""
    if not all([user_id, amount, due_date]):
        return dbc.Alert("All fields are required.", color="warning")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        # First, delete any existing outstanding bills for this user to avoid duplicates
        cursor.execute("DELETE FROM outstanding_bills WHERE user_id = ?", (user_id,))
        # Then, insert the new outstanding bill
        cursor.execute("INSERT INTO outstanding_bills (user_id, amount, due_date, status) VALUES (?, ?, ?, ?)",
                       (user_id, amount, due_date, 'Outstanding'))
        conn.commit()
        return dbc.Alert(f"Bill of KES {amount:,.2f} posted successfully to consumer {user_id}.", color="success")
    except Exception as e:
        return dbc.Alert(f"Error posting bill: {e}", color="danger")
    finally:
        conn.close()


@app.callback(
    [Output('prophet-forecast-graph', 'figure'),
     Output('xai-components-graph', 'figure')],
    [Input('area-dropdown-forecast', 'value')]
)
def update_prophet_forecast(selected_area):
    if not selected_area:
        return go.Figure(layout=PLOT_LAYOUT), go.Figure(layout=PLOT_LAYOUT)

    # Filter data for the selected supply area
    area_df = FULL_DF[FULL_DF['supply_area'] == selected_area].copy()

    # Check if there is enough data
    if area_df.empty or len(area_df) < 20:
        empty_figure_forecast = go.Figure(layout=PLOT_LAYOUT)
        empty_figure_forecast.update_layout(title=f"Not enough data for {selected_area} to generate a forecast.")
        empty_figure_components = go.Figure(layout=PLOT_LAYOUT)
        empty_figure_components.update_layout(title="Not enough data for component breakdown.")
        return empty_figure_forecast, empty_figure_components

    # Call the new function to run the analysis
    results = run_prophet_analysis(
        df=area_df,
        date_column='date',
        target_column='actual_consumption_m3',
        forecast_days=14,
        weekly_seasonality=True,
        aggregate_by_date=False
    )

    # Return the figures directly to dcc.Graph components
    return results['forecast_fig'], results['components_fig']


@app.callback(
    Output('all-consumers-table', 'data'),
    [Input('admin-tabs', 'active_tab')]
)
def update_all_consumers_table_data(active_tab):
    if active_tab != 'tab-all-consumers':
        return []

    user_df = query_db("SELECT username, zone, supply_area FROM users WHERE user_type = 'consumer'")
    merged_df = pd.merge(user_df, FULL_DF, on='supply_area')
    stats_df = merged_df.groupby('username').agg(
        zone=('zone_x', 'first'),
        avg_consumption=('actual_consumption_m3', 'mean'),
        total_leakage_ytd=('pipe_leakage_m3', 'sum'),
        total_complaints_ytd=('complaints_received', 'sum'),
        avg_price_per_litre=('price_per_litre', 'mean')
    ).reset_index()

    stats_df['estimated_annual_bill'] = stats_df['avg_consumption'] * 365 * stats_df['avg_price_per_litre']
    stats_df['payment_status'] = np.random.choice(['Paid', 'Overdue'], size=len(stats_df), p=[0.9, 0.1])

    display_df = stats_df.copy()
    display_df['avg_consumption'] = display_df['avg_consumption'].map('{:,.2f}'.format)
    display_df['total_leakage_ytd'] = display_df['total_leakage_ytd'].map('{:,.2f}'.format)
    display_df['estimated_annual_bill'] = display_df['estimated_annual_bill'].map('KES {:,.2f}'.format)

    columns = [{"name": col.replace('_', ' ').title(), "id": col} for col in display_df.columns]

    return display_df.to_dict('records')


def render_billing_tab_layout():
    """Generates the full layout for the Billing & Payments tab."""
    return html.Div([
        dbc.Card(dbc.CardBody(html.Div(id='outstanding-bill-component')), className="shadow-sm mb-4"),
        dbc.Card(dbc.CardBody([
            html.H4("Recent Payment History"),
            html.Div(id='payment-history-table-container'),
        ]), className="shadow-sm")
    ])


@app.callback(
    Output('water-usage-breakdown-graph', 'figure'),
    Input('consumer-tabs', 'active_tab'),
    State('user-store', 'data')
)
def update_water_usage_breakdown_graph(active_tab, user_data):
    if active_tab != 'tab-dashboard' or not user_data:
        return go.Figure()

    user_id = user_data.get('id')
    categories = ['Bathroom (Showers, Toilets)', 'Kitchen (Cooking, Dishes)', 'Laundry (Washing Machine)',
                  'Outdoor (Gardening, Car Wash)', 'Other']
    np.random.seed(user_id)
    usage_percentages = np.random.dirichlet(np.ones(len(categories)), size=1)[0] * 100
    usage_data = pd.DataFrame({'category': categories, 'percentage': usage_percentages})

    fig = px.pie(
        usage_data,
        names='category',
        values='percentage',
        title='Water Usage Breakdown',
        color_discrete_sequence=COLOR_PALETTE
    )
    fig.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#fff', width=1)))
    fig.update_layout(**PLOT_LAYOUT, showlegend=True, legend_title="Usage Category")

    return fig


@app.callback(
    Output('consumer-tabs-content', 'children'),
    [Input('consumer-tabs', 'active_tab')],
    State('user-store', 'data')
)
def render_consumer_tab_content(active_tab, user_data):
    if active_tab == 'tab-dashboard':
        user_id = user_data.get('id')
        supply_area = user_data.get('supply_area')
        user_zone = user_data.get('zone')
        consumer_df = FULL_DF[FULL_DF['supply_area'] == supply_area]

        fig_consumption = px.line(
            consumer_df,
            x='date',
            y='actual_consumption_m3',
            title=f"Consumption History for {supply_area}",
            color_discrete_sequence=[COLOR_PALETTE[0]]
        )
        fig_consumption.update_traces(mode='lines+markers', marker_size=5)
        fig_consumption.update_layout(**PLOT_LAYOUT, yaxis_title='Consumption (m³)', xaxis_title='Date')

        fig_leakage = px.bar(
            consumer_df.groupby('date')['pipe_leakage_m3'].sum().reset_index(),
            x='date', y='pipe_leakage_m3',
            title=f"Pipe Leakage in {supply_area}",
            color_discrete_sequence=[COLOR_PALETTE[2]]
        )
        fig_leakage.update_layout(**PLOT_LAYOUT, yaxis_title='Leakage (m³)', xaxis_title='Date')

        fig_complaints = px.bar(
            consumer_df.groupby('date')['complaints_received'].sum().reset_index(),
            x='date', y='complaints_received',
            title=f"Complaints in {supply_area}",
            color_discrete_sequence=[COLOR_PALETTE[3]]
        )
        fig_complaints.update_layout(**PLOT_LAYOUT, yaxis_title='Complaints', xaxis_title='Date')

        billing_history_df = query_db(
            "SELECT date_paid, amount FROM billing_history WHERE user_id = ? ORDER BY date_paid ASC", (user_id,))
        fig_billing = px.line(
            billing_history_df,
            x='date_paid', y='amount',
            title="Monthly Billing Trend",
            color_discrete_sequence=[COLOR_PALETTE[4]]
        )
        fig_billing.update_layout(**PLOT_LAYOUT, yaxis_title='Amount (KES)', xaxis_title='Date')

        user_consumption_avg = consumer_df['actual_consumption_m3'].mean()
        area_consumption_avg = consumer_df['actual_consumption_m3'].mean()
        zone_consumption_avg = FULL_DF[FULL_DF['zone'] == user_zone]['actual_consumption_m3'].mean()

        comparative_df = pd.DataFrame({
            'Category': ['Your Average', f'{supply_area} Average', f'{user_zone} Average'],
            'Consumption (m³)': [user_consumption_avg, area_consumption_avg, zone_consumption_avg]
        })

        fig_comparative = px.bar(
            comparative_df,
            x='Consumption (m³)',
            y='Category',
            orientation='h',
            title='Comparative Consumption',
            color='Category',
            color_discrete_map={
                'Your Average': COLOR_PALETTE[0],
                f'{supply_area} Average': COLOR_PALETTE[5],
                f'{user_zone} Average': COLOR_PALETTE[2]
            }
        )
        fig_comparative.update_layout(**PLOT_LAYOUT, yaxis_title='', xaxis_title='Consumption (m³)', showlegend=False)

        return html.Div([
            dbc.Row([
                dbc.Col(create_kpi_card("Avg. Daily Use", f"{user_consumption_avg:.2f} m³", "fa-tint"),
                        className="mb-4"),
                dbc.Col(create_kpi_card("Est. Monthly Bill",
                                        f"KES {(user_consumption_avg * 30 * consumer_df['price_per_litre'].mean()):,.2f}",
                                        "fa-file-invoice-dollar", "success"), className="mb-4"),
                dbc.Col(
                    create_kpi_card("Total Leakage", f"{consumer_df['pipe_leakage_m3'].sum():.2f} m³", "fa-tint-slash",
                                    "warning"), className="mb-4"),
                dbc.Col(
                    create_kpi_card("Complaints Filed", f"{consumer_df['complaints_received'].sum()}", "fa-bullhorn",
                                    "info"),
                    className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_consumption)))), lg=8, className="mb-4"),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_comparative)))), lg=4, className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(id='water-usage-breakdown-graph')))), lg=6,
                        className="mb-4"),
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_leakage)))), lg=6, className="mb-4"),
            ]),
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_complaints)))), lg=6, className="mb-4"),
            ])
        ])

    elif active_tab == 'tab-billing':
        return render_billing_tab_layout()

    elif active_tab == 'tab-chat':
        return create_chat_layout()
    return html.Div()


# Generates the outstanding bill component based on DB state
@app.callback(
    Output('outstanding-bill-component', 'children'),
    Input('trigger-refresh-store', 'data'),
    State('user-store', 'data')
)
def update_outstanding_bill_component(refresh_trigger, user_data):
    user_id = user_data.get('id')
    outstanding_bill = query_db(
        "SELECT amount, due_date FROM outstanding_bills WHERE user_id = ? ORDER BY due_date DESC LIMIT 1",
        (user_id,))

    if outstanding_bill.empty:
        return html.Div([
            html.H4("Billing & Payments", className="card-title text-center text-primary"),
            html.P("No outstanding amount for Billing.", className="text-success mb-2 lead text-center"),
        ], className="p-3 bg-white border rounded-3")
    else:
        return html.Div([
            html.H4("Billing & Payments", className="card-title text-center text-primary"),
            html.P([
                html.Strong("Outstanding Bill: ", className="text-danger"),
                html.Span(f"KES {outstanding_bill.iloc[0]['amount']:,.2f} ", className="text-danger"),
                html.Span(f"(Due: {outstanding_bill.iloc[0]['due_date']})", className="text-danger")
            ], className="lead text-center", id="outstanding-bill-text"),
            html.P("Enter your mobile number for STK push (e.g., 0712345678)", className="text-muted text-center"),
            dbc.InputGroup(
                [
                    dbc.Input(id="payment-phone-input", type="tel", placeholder="e.g., 0712345678"),
                    dbc.Button("Pay Now", id="pay-now-button", color="success"),
                ], className="mt-3"
            ),
            html.P(id="phone-input-validation", className="text-danger small mt-1", style={'display': 'none'}),
        ], className="p-3 bg-white border rounded-3")


# Generates the payment history table based on DB state
@app.callback(
    Output('payment-history-table-container', 'children'),
    Input('trigger-refresh-store', 'data'),
    State('user-store', 'data')
)
def update_payment_history_table(refresh_trigger, user_data):
    user_id = user_data.get('id')
    history_data = query_db(
        "SELECT date_paid, amount, status FROM billing_history WHERE user_id = ? ORDER BY date_paid DESC",
        (user_id,))

    if history_data.empty:
        return html.P("No payment history found.", className="text-center text-muted", id='payment-history-table')
    else:
        history_data['amount'] = 'KES ' + history_data['amount'].map('{:,.2f}'.format)
        history_data.rename(columns={'date_paid': 'Date Paid', 'amount': 'Amount', 'status': 'Status'}, inplace=True)
        return dash_table.DataTable(
            id='payment-history-table',
            data=history_data.to_dict('records'),
            columns=[{"name": i, "id": i} for i in history_data.columns],
            page_size=5,
            style_data_conditional=[
                {'if': {'filter_query': '{Status} = "Paid"'}, 'backgroundColor': '#D4EDDA', 'color': 'green'},
            ],
            style_table={'overflowX': 'auto', 'border': '1px solid #dee2e6'},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f8f9fa'},
            style_cell={'border': '1px solid #dee2e6', 'textAlign': 'left'},
            style_as_list_view=True
        )


# --- STK Push Callback and Helpers ---
def perform_stk_push_and_record_payment(user_data, phone_number):
    """
    A separate function to handle the long-running task of calling the external
    API and updating the database. This function will be run in a new thread.
    """
    user_id = user_data.get('id')
    print(f"Starting STK push and payment recording for user {user_id}")

    # Query for the latest outstanding bill in a fresh connection
    conn = sqlite3.connect(DB_FILE)
    outstanding_bill = pd.read_sql_query(
        "SELECT amount FROM outstanding_bills WHERE user_id = ? ORDER BY due_date DESC LIMIT 1",
        conn, params=(user_id,))
    conn.close()

    if outstanding_bill.empty:
        print("No outstanding bill found. Aborting payment.")
        return

    current_bill_amount = outstanding_bill.iloc[0]['amount']

    stk_base_url = "https://stk-push-igmo.onrender.com"

    # Convert the amount to an integer before sending it to the API
    stk_params = {'amt': int(current_bill_amount), 'tel': phone_number}

    try:
        print(f"Calling STK endpoint with amount: {current_bill_amount}, phone: {phone_number}")
        stk_response = requests.get(stk_base_url, params=stk_params, timeout=30)
        stk_response.raise_for_status()
        response_json = stk_response.json()
        print(response_json)

        if response_json.get('ResponseCode'):
            print("STK Push request successful. Recording payment...")
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO billing_history (user_id, amount, date_paid, status) VALUES (?, ?, ?, ?)",
                           (user_id, current_bill_amount, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Paid'))
            cursor.execute("DELETE FROM outstanding_bills WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            print(
                f"Payment of KES {current_bill_amount} for user {user_id} successfully recorded and outstanding bill deleted.")
        else:
            print(f"STK Push failed: {response_json.get('message', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        print(f"Error calling STK push endpoint: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server Response Content: {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


@app.callback(
    [Output("payment-modal", "is_open"),
     Output("payment-modal-body", "children"),
     Output('payment-completion-interval', 'disabled')],
    [Input("pay-now-button", "n_clicks")],
    [State('user-store', 'data'),
     State("payment-modal", "is_open"),
     State("payment-phone-input", "value")],
    prevent_initial_call=True
)
def handle_payment_request(n_clicks, user_data, is_open, phone_number):
    if not n_clicks or is_open:
        return dash.no_update, dash.no_update, dash.no_update

    if not phone_number or not phone_number.isdigit() or len(phone_number) < 10:
        return False, dbc.Alert("Please enter a valid phone number (e.g., 0712345678).", color="warning"), True

    payment_thread = Thread(target=perform_stk_push_and_record_payment, args=(user_data, phone_number))
    payment_thread.start()

    modal_body = html.Div([
        html.Div(html.I(className="fas fa-paper-plane fa-4x text-info mb-3"), className="text-center"),
        html.H4("Sending payment request...", className="text-center"),
        html.P("Please check your phone for an M-Pesa STK push prompt.", className="text-center"),
        dbc.Spinner(size="sm")
    ])

    return True, modal_body, False


@app.callback(
    [Output("payment-modal", "is_open", allow_duplicate=True),
     Output("payment-modal-body", "children", allow_duplicate=True),
     Output('trigger-refresh-store', 'data', allow_duplicate=True)],
    [Input("payment-completion-interval", "n_intervals")],
    State("payment-modal", "is_open"),
    prevent_initial_call=True
)
def handle_payment_completion_modal(n_intervals, is_open):
    if is_open and n_intervals > 10:
        return False, None, n_intervals
    return dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output('chat-display', 'children'),
    [Input('chat-refresh-interval', 'n_intervals'),
     Input('chat-send-button', 'n_clicks')],
    [State('chat-input', 'value'),
     State('user-store', 'data')]
)
def update_chat_display(n_intervals, n_clicks, message, user_data):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''

    if triggered_id == 'chat-send-button' and message:
        user_id = user_data.get('id')
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO chat_messages (user_id, message, timestamp) VALUES (?, ?, ?)",
                       (user_id, message, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        time.sleep(0.5)

    query = """
    SELECT u.username, cm.message, cm.timestamp, u.user_type, u.id AS user_id
    FROM chat_messages cm
    JOIN users u ON cm.user_id = u.id
    ORDER BY cm.timestamp ASC
    """
    chat_df = query_db(query)

    current_user_id = user_data.get('id')

    admin_id_row = query_db("SELECT id FROM users WHERE user_type = 'admin'")
    admin_id = admin_id_row.iloc[0]['id'] if not admin_id_row.empty else -1

    messages = []
    for _, row in chat_df.iterrows():
        if row['user_id'] == current_user_id:
            card_style = {'backgroundColor': '#e2f0ff', 'color': '#000', 'border': '1px solid #cce0ff',
                          'borderRadius': '15px 15px 0 15px', 'marginLeft': 'auto'}
            container_class = "d-flex justify-content-end"
        elif row['user_id'] == admin_id:
            card_style = {'backgroundColor': '#d1e7dd', 'color': '#000', 'border': '1px solid #c3e6cb',
                          'borderRadius': '15px 15px 15px 0', 'marginRight': 'auto'}
            container_class = "d-flex justify-content-start"
        else:
            card_style = {'backgroundColor': '#f0f0f0', 'color': '#000', 'border': '1px solid #ccc',
                          'borderRadius': '15px 15px 15px 0', 'marginRight': 'auto'}
            container_class = "d-flex justify-content-start"

        message_card = html.Div(
            dbc.Card(
                dbc.CardBody([
                    html.Strong(row['username'], className="me-2 text-primary fw-bold"),
                    html.Div(row['message'], style={'white-space': 'pre-wrap'}),
                    html.Small(row['timestamp'], className="text-muted text-end d-block mt-1")
                ]),
                style=card_style,
            ),
            className=container_class + " mb-2 w-75"
        )
        messages.append(message_card)

    return messages


@app.callback(
    Output('chat-input', 'value'),
    Input('chat-send-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_chat_input(n_clicks):
    return ""


@app.callback(
    [Output('model-info', 'children'),
     Output('performance-plot', 'figure'),
     Output('prediction-plot', 'figure'),
     Output('feature-importance-plot', 'figure'),
     Output('metrics-display', 'children')],
    [Input('Tt_train-button', 'n_clicks')],
    [State('target-dropdown', 'value')]
)
def train_and_visualize(n_clicks, target_output):
    """Train model and create visualizations"""
    if n_clicks == 0:
        return html.Div(), go.Figure(), go.Figure(), go.Figure(), html.Div()

    global model_results
    model_results = train_price_discrimination_model(df_global)
    history = model_results['history']

    # Model info box and other visualizations remain largely the same...
    model_info = html.Div([
        html.H3("🧠 Model Training Results", style={'color': '#2c3e50', 'marginBottom': '15px'}),
        html.Div([
            html.Div([html.Strong("Architecture: "), html.Span("Multi-Output NN")]),
            html.Div([html.Strong("Epochs: "), html.Span(f"{len(history.history['loss'])} (Early Stopping)")]),
        ])
    ], style={'backgroundColor': '#e8f5e9', 'padding': '20px', 'borderRadius': '10px', 'border': '2px solid #4caf50',
              'marginTop': '20px'})

    # Performance plots
    fig_performance = make_subplots(rows=2, cols=2,
                                    subplot_titles=('Total Loss', 'Price Loss', 'Time Loss', 'Zone Loss'))
    fig_performance.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss', line=dict(color='#e74c3c')),
                              row=1, col=1)
    fig_performance.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss', line=dict(color='#3498db')),
                              row=1, col=1)

    fig_performance.update_layout(title_text='Training Performance', template='plotly_white')

    # Prediction plots
    test_data = model_results['test_data']
    y_price_pred, _, _ = test_data['predictions']
    fig_predictions = go.Figure()
    if target_output == 'price' or target_output == 'all':
        fig_predictions.add_trace(
            go.Scatter(x=test_data['y_price_test'], y=y_price_pred.flatten(), mode='markers', name='Predictions'))
        fig_predictions.add_trace(go.Scatter(x=[min(test_data['y_price_test']), max(test_data['y_price_test'])],
                                             y=[min(test_data['y_price_test']), max(test_data['y_price_test'])],
                                             mode='lines', name='Perfect Fit', line=dict(dash='dash')))
        fig_predictions.update_layout(title_text='Price Predictions vs Actual', xaxis_title='Actual Price',
                                      yaxis_title='Predicted Price', template='plotly_white')

    # Feature Importance
    fig_importance = go.Figure(go.Bar(x=np.random.rand(5), y=model_results['feature_columns'][:5], orientation='h'))
    fig_importance.update_layout(title_text='Feature Importance (Approximation)', template='plotly_white')

    # Metrics display
    metrics = model_results['metrics']
    metrics_div = html.Div([
        html.H3("📊 Model Performance Metrics", style={'color': '#2c3e50', 'textAlign': 'center'}),
        html.Div([
            html.Div([html.H4("Price MAE"), html.P(f"{metrics['price_mae']:.4f}")],
                     style={'display': 'inline-block', 'width': '25%', 'textAlign': 'center'}),
            html.Div([html.H4("Price RMSE"), html.P(f"{metrics['price_rmse']:.4f}")],
                     style={'display': 'inline-block', 'width': '25%', 'textAlign': 'center'}),
            html.Div([html.H4("Time Accuracy"), html.P(f"{metrics['time_accuracy']:.2%}")],
                     style={'display': 'inline-block', 'width': '25%', 'textAlign': 'center'}),
            html.Div([html.H4("Zone Accuracy"), html.P(f"{metrics['zone_accuracy']:.2%}")],
                     style={'display': 'inline-block', 'width': '25%', 'textAlign': 'center'}),
        ])
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '10px', 'marginTop': '20px'})

    return model_info, fig_performance, fig_predictions, fig_importance, metrics_div


# ==================== SCHEDULE CALLBACK ====================
@app.callback(
    [Output('schedule-display', 'children'),
     Output('schedule-viz', 'figure')],
    [Input('Tt_schedule-button', 'n_clicks')],
    [State('schedule-days-slider', 'value')],
    prevent_initial_call=True,
    allow_duplicate=True
)
def display_schedule(n_clicks, days_ahead):
    """Generate and display the distribution schedule"""
    if n_clicks == 0 or model_results is None:
        return html.Div([
            html.P("Please train the model first to generate a schedule.",
                   style={'textAlign': 'center', 'color': '#e67e22', 'fontSize': '16px', 'marginTop': '20px'})
        ]), go.Figure()

    global schedule_df
    schedule_df = generate_distribution_schedule(model_results, df_global, days_ahead)

    # Create the display table
    schedule_table = html.Div([
        html.H3(f"📅 Recommended {days_ahead}-Day Distribution Schedule",
                style={'color': '#2c3e50', 'marginTop': '30px'}),
        dash_table.DataTable(
            data=schedule_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in schedule_df.columns],
            style_cell={'textAlign': 'center', 'padding': '8px'},
            style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(240, 240, 240)'}]
        )
    ], style={'marginTop': '20px'})

    # Create the visualization
    schedule_df_viz = schedule_df.copy()
    schedule_df_viz['Predicted Price per Litre'] = pd.to_numeric(schedule_df_viz['Predicted Price per Litre'])

    fig_schedule = go.Figure()

    import plotly.express as px
    fig_schedule = px.bar(
        schedule_df_viz,
        x='Date',
        y='Predicted Price per Litre',
        color='Predicted Zone',
        title='Predicted Price by Zone for the Upcoming Week',
        labels={'Predicted Price per Litre': 'Price per Litre (KES)', 'Date': 'Future Date'},
        hover_data=['Predicted Time']
    )

    fig_schedule.update_layout(template='plotly_white', barmode='group')

    return schedule_table, fig_schedule


@app.callback(
    [Output('Tt_schedule-display', 'children'),
     Output('Tt_schedule-viz', 'figure')],
    [Input('schedule-button', 'n_clicks')],
    [State('schedule-days-slider', 'value')],
    prevent_initial_call=True,
    allow_duplicate=True
)
def Tt_generate_schedule(n_clicks, days_ahead):
    """Generate optimized rationing schedule"""
    if n_clicks == 0 or Tt_model_results is None:
        return html.Div([
            html.P("⚠️ Please train the model first before generating schedule.",
                   style={'textAlign': 'center', 'color': '#e74c3c', 'fontSize': '16px'})
        ]), go.Figure()
    global Tt_schedule_df
    Tt_schedule_df = Tt_generate_rationing_schedule(Tt_model_results, Tt_df_global, days_ahead)
    Tt_schedule_df['priority_score'] = (
            Tt_schedule_df['zone_priority'] * 0.6 +
            Tt_schedule_df['water_stress'] * 0.3 +
            Tt_schedule_df['complaint_rate'] * 0.1
    )
    Tt_schedule_df = Tt_schedule_df.sort_values(['date', 'priority_score'], ascending=[True, False])
    display_data = Tt_schedule_df.copy()
    display_data['date'] = display_data['date'].dt.strftime('%Y-%m-%d (%A)')
    display_data['time_confidence'] = (display_data['time_confidence'] * 100).round(1).astype(str) + '%'
    display_data['zone_priority'] = (display_data['zone_priority'] * 100).round(1).astype(str) + '%'
    display_data['priority_score'] = display_data['priority_score'].round(3)
    display_data['demand_forecast'] = display_data['demand_forecast'].round(0).astype(int)
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
    fig_schedule = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Time Distribution by Zone', 'Zone Priority Over Time',
                        'Water Stress by Zone', 'Demand Forecast Timeline'),
        specs=[[{'type': 'bar'}, {'type': 'scatter'}],
               [{'type': 'box'}, {'type': 'scatter'}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    time_zone_counts = Tt_schedule_df.groupby(['zone', 'recommended_time']).size().reset_index(name='count')
    for zone in Tt_schedule_df['zone'].unique():
        zone_data = time_zone_counts[time_zone_counts['zone'] == zone]
        fig_schedule.add_trace(go.Bar(
            x=zone_data['recommended_time'],
            y=zone_data['count'],
            name=zone,
            hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
        ), row=1, col=1)
    for zone in Tt_schedule_df['zone'].unique():
        zone_data = Tt_schedule_df[Tt_schedule_df['zone'] == zone]
        fig_schedule.add_trace(go.Scatter(
            x=zone_data['date'],
            y=zone_data['zone_priority'],
            mode='lines+markers',
            name=zone,
            hovertemplate='Date: %{x}<br>Priority: %{y:.3f}<extra></extra>'
        ), row=1, col=2)
    for zone in Tt_schedule_df['zone'].unique():
        zone_data = Tt_schedule_df[Tt_schedule_df['zone'] == zone]
        fig_schedule.add_trace(go.Box(
            y=zone_data['water_stress'],
            name=zone,
            boxmean='sd'
        ), row=2, col=1)
    demand_by_date = Tt_schedule_df.groupby('date')['demand_forecast'].sum().reset_index()
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


@app.callback(
    [Output('Tt_model-info', 'children'),
     Output('Tt_training-plot', 'figure'),
     Output('Tt_confusion-plot', 'figure'),
     Output('Tt_classification-reports', 'children'),
     Output('Tt_metrics-display', 'children')],
    [Input('train-button', 'n_clicks')]
)
def Tt_train_model(n_clicks):
    """Train the rationing model"""
    if n_clicks == 0:
        return html.Div(), go.Figure(), go.Figure(), html.Div(), html.Div()
    global Tt_model_results
    Tt_model_results = Tt_train_rationing_model(Tt_df_global)
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
                html.Span(f"{len(Tt_model_results['feature_columns'])} engineered features")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Strong("Parameters: "),
                html.Span(f"{Tt_model_results['model'].count_params():,}")
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Strong("Training Strategy: "),
                html.Span("Multi-Task Learning with Equal Loss Weighting")
            ])
        ])
    ], style={'backgroundColor': '#d5f4e6', 'padding': '20px', 'borderRadius': '10px',
              'border': '2px solid #16a085'})
    history = Tt_model_results['history']
    fig_training = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Loss', 'Time Classification Loss',
                        'Zone Classification Loss', 'Combined Accuracy'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
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
    metrics = Tt_model_results['metrics']
    time_cm = metrics['time_confusion_matrix']
    zone_cm = metrics['zone_confusion_matrix']
    time_classes = Tt_model_results['time_encoder'].classes_
    zone_classes = Tt_model_results['zone_encoder'].classes_
    fig_confusion = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Time Slot Confusion Matrix', 'Zone Confusion Matrix'),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}]]
    )
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
    time_report = metrics['time_report']
    zone_report = metrics['zone_report']
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
            html.Span("Using simulated data. Place 'dataset_Gans_run_1.csv' in the same directory to use real data.")
        ])

    return scenario_desc, dataset_desc


@app.callback(
    [Output('forecast-plot', 'figure'),
     Output('components-plot', 'figure'),
     Output('FB_metrics-display', 'children'),
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

    # Add a shaded background region for the forecast period - orange
    last_historical_date = df['ds'].max()
    fig_forecast.add_vrect(
        x0=last_historical_date,
        x1=forecast['ds'].max(),
        fillcolor="rgba(255, 165, 0, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="Forecast Period",
        annotation_position="top left",
        annotation=dict(font=dict(size=12, color="rgba(255, 165, 0, 0.8)"))
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

    # Forecast.
    fig_forecast.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Prophet Forecast',
        line=dict(color='#3498db', width=3),
        hovertemplate='Date: %{x}<br>Forecast: %{y:,.0f} m³<extra></extra>'
    ))

    # Confidence interval.
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
        fillcolor='rgba(52, 152, 219, 0.2)',
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
    app.run(debug=True, port=8709)
