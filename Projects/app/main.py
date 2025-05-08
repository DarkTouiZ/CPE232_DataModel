import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from prophet import Prophet
from prophet.serialize import model_from_json
from catboost import CatBoostRegressor
import openmeteo_requests
import requests_cache
from retry_requests import retry
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="การเกษตรกร",
    page_icon="🧙🏻",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E3A8A;
    }
    .pm-safe {color: #10B981;}
    .pm-moderate {color: #FBBF24;}
    .pm-unhealthy {color: #EF4444;}
    .stMetric {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        with open('app/models/39T/prophet_model_39T.json', 'r') as f:
            prophet_model = model_from_json(f.read())
        catboost_model = CatBoostRegressor().load_model('app/models/39T/residual_model_39T.cbm')
        scaler = joblib.load('app/models/39T/scaler_39T.pkl')
        with open('app/models/39T/selected_features_39T.pkl', 'rb') as f:
            selected_features = pickle.load(f)
        return prophet_model, catboost_model, scaler, selected_features
    except Exception as e:
        st.error(f"Error loading models: {e}")

# Load models
prophet_model, catboost_model, scaler, selected_features = load_models()

@st.cache_resource
def load_historical_PM25():
    try:
        historical_pm25 = pd.read_csv('dataset/PM2.5/1m_history_PM25.csv')
        historical_pm25['date'] = pd.to_datetime(historical_pm25['date'])
        return historical_pm25
    except Exception as e:
        st.error(f"Error loading historical PM2.5 data: {e}")

hist_PM25_df = load_historical_PM25()

# Define location options
LOCATIONS = {
    "สำนักงานสาธารณสุขอำเภอแม่สาย": {"lat": 20.4275, "lon": 99.8836},
    "โครงการชลประทานนครสวรรค์": {"lat": 15.6931, "lon": 100.1364},
    "โรงพยาบาลเฉลิมพระเกียรติ น่าน": {"lat": 19.5757, "lon": 101.0821},
    "สำนักงานทรัพยากรธรรมชาติและสิ่งแวดล้อมจังหวัดแม่ฮ่องสอน": {"lat": 19.3014, "lon": 97.9708},
    "โรงพยาบาลส่งเสริมสุขภาพตำบลท่าสี": {"lat": 18.4272, "lon": 99.7577},
    "ศูนย์การศึกษานอกระบบอำเภอแม่สอด": {"lat": 16.7465, "lon": 98.5744}
}
STATION_ID2NAME = {
    "73T": "สำนักงานสาธารณสุขอำเภอแม่สาย",
    "41T": "โครงการชลประทานนครสวรรค์",
    "75T": "โรงพยาบาลเฉลิมพระเกียรติ น่าน",
    "58T": "สำนักงานทรัพยากรธรรมชาติและสิ่งแวดล้อมจังหวัดแม่ฮ่องสอน",
    "39T": "โรงพยาบาลส่งเสริมสุขภาพตำบลท่าสี",
    "76T": "ศูนย์การศึกษานอกระบบอำเภอแม่สอด"
}
STATION_NAME2ID = {v: k for k, v in STATION_ID2NAME.items()}

# Fetch weather data from Open-Meteo
@st.cache_data(ttl=3600)
def fetch_weather_data(lat, lon, days=16):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["weather_code", "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
                 "precipitation_sum", "rain_sum", "showers_sum",
                 "wind_speed_10m_mean", "wind_direction_10m_dominant", 
                 "cloud_cover_mean", "relative_humidity_2m_mean", "pressure_msl_mean"],
        "timezone": "auto",
        "forecast_days": days
    }
    
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        daily = response.Daily()
        daily_data = {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s"),
                periods=len(daily.Variables(0).ValuesAsNumpy()),
                freq="D"
            ),
            "weather_code": daily.Variables(0).ValuesAsNumpy(),
            "temperature_2m_mean": daily.Variables(1).ValuesAsNumpy(),
            "temperature_2m_min": daily.Variables(2).ValuesAsNumpy(),
            "temperature_2m_max": daily.Variables(3).ValuesAsNumpy(),
            "precipitation_sum": daily.Variables(4).ValuesAsNumpy(),
            "rain_sum": daily.Variables(5).ValuesAsNumpy(),
            "showers_sum": daily.Variables(6).ValuesAsNumpy(),
            "wind_speed_10m_mean": daily.Variables(7).ValuesAsNumpy(),
            "wind_direction_10m_dominant": daily.Variables(8).ValuesAsNumpy(),
            "cloud_cover_mean": daily.Variables(9).ValuesAsNumpy(),
            "relative_humidity_2m_mean": daily.Variables(10).ValuesAsNumpy(),
            "pressure_msl_mean": daily.Variables(11).ValuesAsNumpy(),
            "fire_count_total": np.random.poisson(lam=2, size=days)  # Simulated fire count data; replace with actual API
        }
        return pd.DataFrame(data=daily_data)
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")

# Feature engineering function
def create_features(df):
    df = df.copy()

    # Ensure datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Calendar features
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear

    # Seasonal features
    df['season'] = pd.cut(
        df['date'].dt.month,
        bins=[0, 2, 5, 10, 12],
        labels=['Cool', 'Summer', 'Rainy', 'Cool'],
        ordered=False
    )

    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['dayofyear']/365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['dayofyear']/365)

    # Lagged and rolling features for pm25 (only where pm25 is available)
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df[f'pm25_lag_{lag}'] = df['pm25'].shift(lag)
    for window in [7, 14, 30]:
        df[f'pm25_rolling_mean_{window}'] = df['pm25'].rolling(window=window, min_periods=1).mean()
        df[f'pm25_rolling_std_{window}'] = df['pm25'].rolling(window=window, min_periods=1).std()
        df[f'pm25_rolling_max_{window}'] = df['pm25'].rolling(window=window, min_periods=1).max()
        df[f'pm25_rolling_min_{window}'] = df['pm25'].rolling(window=window, min_periods=1).min()
    
    # Lagged and rolling for important variables
    important_cols = [
        'fire_count_total',
        'relative_humidity_2m_mean (%)',
        'temperature_2m_mean (°C)_center',
        'precipitation_sum (mm)_center',
        'cloud_cover_mean (%)'
    ]
    for col in important_cols:
        if col in df.columns:
            for lag in [1, 3, 7, 14]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            for window in [7, 14]:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()

    # Expanding features
    df['pm25_expanding_mean'] = df['pm25'].expanding(min_periods=1).mean()
    df['fire_count_expanding_sum'] = df['fire_count_total'].expanding(min_periods=1).sum()

    # Diff features
    df['pm25_diff_1'] = df['pm25'].diff(1)
    df['temperature_diff_1'] = df['temperature_2m_mean (°C)_center'].diff(1)
    df['humidity_diff_1'] = df['relative_humidity_2m_mean (%)'].diff(1)

    # Aggregate features across regions
    df['temperature_mean_all_regions'] = df[[
        'temperature_2m_mean (°C)_north',
        'temperature_2m_mean (°C)_south',
        'temperature_2m_mean (°C)_east',
        'temperature_2m_mean (°C)_west'
    ]].mean(axis=1)

    df['precipitation_sum_all_regions'] = df[[
        'precipitation_sum (mm)_north',
        'precipitation_sum (mm)_south',
        'precipitation_sum (mm)_east',
        'precipitation_sum (mm)_west'
    ]].sum(axis=1)

    # Wind direction decomposition
    df['wind_x'] = np.cos(np.radians(df['wind_direction_10m_dominant (°)']))
    df['wind_y'] = np.sin(np.radians(df['wind_direction_10m_dominant (°)']))

    # Weather code encoding
    rainy_codes = [51, 61, 63, 65]
    df['is_rainy_weather'] = df['weather_code (wmo code)'].isin(rainy_codes).astype(int)

    # Interaction features
    df['temp_humidity'] = df['temperature_2m_mean (°C)_center'] * df['relative_humidity_2m_mean (%)']
    df['wind_temp'] = df['wind_speed_10m_mean (km/h)'] * df['temperature_2m_mean (°C)_center']
    df['fire_temp'] = df['fire_count_total'] * df['temperature_2m_mean (°C)_center']
    df['fire_humidity'] = df['fire_count_total'] * df['relative_humidity_2m_mean (%)']

    # Binary thresholds
    df['is_march'] = (df['date'].dt.month == 3).astype(int)
    df['is_dry_season'] = df['season'].isin(['Cool', 'Summer']).astype(int)
    df['is_burning_season'] = df['date'].dt.month.isin([1, 2, 3, 4]).astype(int)
    df['is_high_temp'] = (df['temperature_2m_mean (°C)_center'] > 33).astype(int)
    df['is_low_humidity'] = (df['relative_humidity_2m_mean (%)'] < 55).astype(int)

    # Moving average of lagged targets
    df['pm25_lag_1_3_mean'] = (df['pm25'].shift(1) + df['pm25'].shift(2) + df['pm25'].shift(3)) / 3

    # Encode season
    df['season'] = df['season'].map({'Cool': 0, 'Summer': 1, 'Rainy': 2})

    # Handle missing values for non-forecast features
    for col in df.columns:
        if col != 'pm25' and df[col].dtype != 'object' and df[col].dtype != 'category':
            df[col] = df[col].fillna(df[col].mean())
    
    return df

# Prediction function
def predict_pm25(df_input, location_name):
    # Apply feature engineering
    df_features = create_features(df_input)
    
    # Extract ds for Prophet (making sure to remove timezone info)
    prophet_df = pd.DataFrame({'ds': df_features['date'].dt.tz_localize(None), 'y': 0})
    
    # Prophet predictions (timezone-free)
    prophet_forecast = prophet_model.predict(prophet_df)
    
    # Add the prophet predictions back to the features dataframe
    df_features['prophet_pred'] = prophet_forecast['yhat'].values
    df_features['prophet_trend'] = prophet_forecast['trend'].values
    df_features['prophet_lower'] = prophet_forecast['yhat_lower'].values
    df_features['prophet_upper'] = prophet_forecast['yhat_upper'].values
    
    # Prepare features for CatBoost
    drop_cols = ['date', 'season', 'prophet_lower', 'prophet_upper']
    X = df_features.drop(columns=drop_cols, errors='ignore')
    
    # Feature selection
    if len(selected_features) > 0:
        available_features = [f for f in selected_features if f in X.columns]
        X = X[available_features]

    
    # CatBoost residual predictions
    try:
        cat_residual_preds = catboost_model.predict(X)
        hybrid_preds = df_features['prophet_pred'].values + cat_residual_preds
    except:
        hybrid_preds = df_features['prophet_pred'].values
        
    # Apply location-specific and burning season factors
    hybrid_preds = hybrid_preds
    
    # Return predictions and dates
    result = {
        'date': df_features['date'],
        'pm25_pred': hybrid_preds,
        'prophet_lower': df_features['prophet_lower'],
        'prophet_upper': df_features['prophet_upper'],
        'season': df_features['season'],
        'is_burning_season': df_features['is_burning_season']
    }
    
    return pd.DataFrame(result)

def get_pm25_category(value):
    if value <= 25:
        return "ดีมาก", "#10B981", "ปลอดภัย"
    elif value <= 37:
        return "ดี", "#FBBF24", "ปานกลาง"
    elif value <= 50:
        return "ปานกลาง", "#F59E0B", "ไม่ดีสำหรับกลุ่มเสี่ยง"
    elif value <= 90:
        return "เริ่มมีผลกระทบต่อสุขภาพ", "#EF4444", "ไม่ดีต่อสุขภาพ"
    else:
        return "มีผลกระทบต่อสุขภาพ", "#581C87", "อันตรายมาก"

def plot_pm25_forecast(df_result):
    # Create color mapping based on PM2.5 levels
    colors = []
    tooltips = []
    
    for val in df_result['pm25_pred']:
        category, color, thai_cat = get_pm25_category(val)
        colors.append(color)
        tooltips.append(f"{category} ({thai_cat})")
    
    # Create the figure
    fig = go.Figure()
    
    # Add prediction line
    fig.add_trace(go.Scatter(
        x=df_result['date'], 
        y=df_result['pm25_pred'],
        mode='lines+markers',
        name='PM2.5 Forecast',
        line=dict(width=3),
        marker=dict(color=colors, size=10),
        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                      '<b>PM2.5:</b> %{y:.1f} μg/m³<br>' +
                      '<b>Category:</b> %{text}<extra></extra>',
        text=tooltips
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([df_result['date'], df_result['date'].iloc[::-1]]),
        y=pd.concat([df_result['prophet_upper'], df_result['prophet_lower'].iloc[::-1]]),
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    fig.add_hline(y=25, line_dash="dash", line_color="#10B981", 
                  annotation_text="ดีมาก", annotation_position="top right")
    fig.add_hline(y=37, line_dash="dash", line_color="#FBBF24", 
                  annotation_text="ดี", annotation_position="top right")  
    fig.add_hline(y=50, line_dash="dash", line_color="#F59E0B", 
                  annotation_text="ปานกลาง", annotation_position="top right")
    fig.add_hline(y=90, line_dash="dash", line_color="#EF4444", 
                  annotation_text="เริ่มมีผลกระทบต่อสุขภาพ", annotation_position="top right")
    
    # Layout configuration
    fig.update_layout(
        title='PM2.5 Forecast with Confidence Interval',
        xaxis_title='Date',
        yaxis_title='PM2.5 (μg/m³)',
        hovermode='x unified',
        legend_title='Legend',
        template='plotly_white',
        height=600,
    )
    
    # Set y-axis to start from 0
    fig.update_yaxes(rangemode='tozero')
    
    return fig

# Main Streamlit UI
st.title("PM2.5 Forecast")
st.write("This application forecasts PM2.5 levels using a hybrid Prophet + CatBoost model")

# Sidebar for controls
with st.sidebar:
    location = st.selectbox(
        "Select Location", 
        options=list(LOCATIONS.keys()),
        index=0
    )
    
    forecast_days = st.slider(
        "Forecast Days", 
        min_value=7, 
        max_value=16, 
        value=16
    )

# Get data and make predictions
lat, lon = LOCATIONS[location]["lat"], LOCATIONS[location]["lon"]
df_weather = fetch_weather_data(lat, lon, days=forecast_days)

# Prepare historical PM2.5 data
station_id = STATION_NAME2ID[location].lower()
hist_PM25 = hist_PM25_df[['date', station_id]].rename(columns={station_id: 'pm25'})

# Rename weather columns to match expected names
df_weather = df_weather.rename(columns={
    'temperature_2m_mean': 'temperature_2m_mean (°C)_center',
    'temperature_2m_min': 'temperature_2m_min (°C)',
    'temperature_2m_max': 'temperature_2m_max (°C)',
    'precipitation_sum': 'precipitation_sum (mm)_center',
    'rain_sum': 'rain_sum (mm)',
    'showers_sum': 'showers_sum (mm)',
    'wind_speed_10m_mean': 'wind_speed_10m_mean (km/h)',
    'wind_direction_10m_dominant': 'wind_direction_10m_dominant (°)',
    'cloud_cover_mean': 'cloud_cover_mean (%)',
    'relative_humidity_2m_mean': 'relative_humidity_2m_mean (%)',
    'pressure_msl_mean': 'pressure_msl_mean (hPa)',
    'weather_code': 'weather_code (wmo code)'
})

# Add placeholder columns for regional data
for region in ['north', 'south', 'east', 'west']:
    df_weather[f'temperature_2m_mean (°C)_{region}'] = df_weather['temperature_2m_mean (°C)_center']
    df_weather[f'precipitation_sum (mm)_{region}'] = df_weather['precipitation_sum (mm)_center']

# Combine historical PM2.5 and forecast weather data
df_weather['pm25'] = np.nan  # No PM2.5 for forecast period
df_input = pd.concat([hist_PM25, df_weather], ignore_index=True, sort=False)

# Ensure date is datetime and sorted
df_input['date'] = pd.to_datetime(df_input['date'])
df_input = df_input.sort_values('date').reset_index(drop=True)

# Fill missing weather data in historical period with mean values
weather_cols = [
    'temperature_2m_mean (°C)_center', 'temperature_2m_min (°C)', 'temperature_2m_max (°C)',
    'precipitation_sum (mm)_center', 'rain_sum (mm)', 'showers_sum (mm)',
    'wind_speed_10m_mean (km/h)', 'wind_direction_10m_dominant (°)',
    'cloud_cover_mean (%)', 'relative_humidity_2m_mean (%)', 'pressure_msl_mean (hPa)',
    'weather_code (wmo code)', 'fire_count_total'
] + [f'temperature_2m_mean (°C)_{r}' for r in ['north', 'south', 'east', 'west']] \
  + [f'precipitation_sum (mm)_{r}' for r in ['north', 'south', 'east', 'west']]

for col in weather_cols:
    if col not in df_input.columns:
        df_input[col] = np.nan
    df_input[col] = df_input[col].fillna(df_input[col].mean())

# Proceed with prediction
result_df = predict_pm25(df_input, location)

# Display forecast chart
st.plotly_chart(plot_pm25_forecast(result_df), use_container_width=True)

# Display data table
st.subheader("Forecast Data")
display_df = result_df.copy()
display_df['Date'] = display_df['date'].dt.strftime('%Y-%m-%d')
display_df['PM2.5 Forecast (μg/m³)'] = display_df['pm25_pred'].round(1)
display_df['Range'] = display_df.apply(
    lambda x: f"{round(x['prophet_lower'], 1)} - {round(x['prophet_upper'], 1)}", axis=1
)
display_df['Category'] = display_df['pm25_pred'].apply(
    lambda x: get_pm25_category(x)[0]
)

# Show only necessary columns
st.dataframe(display_df[['Date', 'PM2.5 Forecast (μg/m³)', 'Range', 'Category']], hide_index=True)