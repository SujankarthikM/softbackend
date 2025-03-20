from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.utils import CustomObjectScope
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU-only mode
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress unnecessary TensorFlow warnings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# API key for scrapingdog
API_KEY = "67dc04c3f7277221019e339b"
SCRAPINGDOG_URL = "https://api.scrapingdog.com/google"


# Utility function to fetch AQI data
def fetch_aqi_data(query):
    params = {
        "api_key": API_KEY,
        "query": f"{query} air quality index",
        "results": 10,
        "country": "in",
        "page": 0,
        "advance_search": "false"
    }

    response = requests.get(SCRAPINGDOG_URL, params=params)
    if response.status_code != 200:
        return {"error": "Failed to fetch data from Google"}, 500

    data = response.json()
    links = []

    # Extract AQI-related links
    if 'peopleAlsoAskedFor' in data:
        links.extend(
            item['link']
            for item in data['peopleAlsoAskedFor']
            if 'link' in item and 'aqi.in' in item['link']
        )
    if 'organic_results' in data:
        links.extend(
            item['link']
            for item in data['organic_results']
            if 'link' in item and 'aqi.in' in item['link']
        )

    if not links:
        return {"error": "No valid AQI links found"}, 404

    # Fetch AQI page
    aqi_url = links[0]
    response = requests.get(aqi_url)
    if response.status_code != 200:
        return {"error": "Failed to fetch AQI page"}, 500

    # Extract and clean AQI HTML
    raw_html = response.text
    cleaned_html = ''
    for i in range(len(raw_html)):
        cleaned_html += raw_html[i]
        if raw_html[i:i + 41] == "rank-comparison px-body w-auto sm:mx-body":
            break
    cleaned_html = cleaned_html[:-13] + '</body></html>'

    # Further cleanup
    result_html = ''
    i = 0
    while i < len(cleaned_html):
        if cleaned_html[i:i + 34] == 'z-[999] bg-white dark:bg-[#22272C]':
            while cleaned_html[i:i + 19] != '<span>Last Updated:' and i + 19 <= len(cleaned_html):
                i += 1
        result_html += cleaned_html[i]
        i += 1

    soup = BeautifulSoup(result_html, 'html.parser')
    return {"html": str(soup)}


# Utility function to fetch weather data
def fetch_weather_data(query):
    params = {
        "api_key": API_KEY,
        "query": f"{query} past weather data timeanddate.com",
        "results": 10,
        "country": "in",
        "page": 0,
        "advance_search": "false"
    }

    response = requests.get(SCRAPINGDOG_URL, params=params)
    if response.status_code != 200:
        return {"error": "Failed to fetch data from Google"}, 500

    data = response.json()
    links = []

    # Extract weather-related links
    if 'organic_results' in data:
        links.extend(
            item['link']
            for item in data['organic_results']
            if 'link' in item and 'timeanddate.com' in item['link']
        )

    if not links:
        return {"error": "No valid weather links found"}, 404

    # Fetch weather page
    weather_url = links[0]
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(weather_url, headers=headers)
    if response.status_code != 200:
        return {"error": "Failed to fetch weather page"}, 500

    # Extract weather data
    raw_html = response.text
    data_section = ''
    for i in range(len(raw_html)):
        if raw_html[i:i + 9] == 'var data=':
            while raw_html[i] != ';':
                data_section += raw_html[i]
                i += 1
            break

    data_section = data_section[9:]
    data_section = data_section.replace('true', 'True').replace('\\/', '/')

    try:
        weather_data = eval(data_section)['detail']
    except Exception as e:
        return {"error": f"Failed to parse weather data: {str(e)}"}, 500

    # Process weather data into DataFrame
    df = pd.DataFrame(weather_data)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df = df[['date', 'ts', 'ds', 'desc', 'temp', 'templow', 'baro', 'wind', 'wd', 'hum']]
    return df.to_dict(orient='records')


# Weather prediction functions
def get_city_coordinates(city_name, api_key):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
    response = requests.get(url)
    data = response.json()
    if data:
        return data[0]['lat'], data[0]['lon']
    else:
        return None, None

def fetch_visual_crossing_data(latitude, longitude, start_date, end_date, api_key):
    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    location = f"{latitude},{longitude}"
    url = f"{base_url}/{location}/{start_date}/{end_date}"
    params = {
        "unitGroup": "metric",
        "include": "hours",
        "key": api_key,
        "elements": "datetime,temp,humidity,pressure,cloudcover,conditions,windspeed",
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        weather_data = []
        for day in data['days']:
            for hour in day['hours']:
                weather_data.append([
                    f"{day['datetime']}T{hour['datetime']}",
                    hour['temp'],
                    hour['humidity'],
                    hour['pressure'],
                    hour['cloudcover'],
                    1 if hour['conditions'] == "Clear" else 0,
                    hour['windspeed'],
                ])
        df = pd.DataFrame(weather_data, columns=[
            'datetime', 'temperature', 'humidity', 'pressure', 'cloud_cover', 'weather_code', 'wind_speed'
        ])
        return df
    else:
        return None

def load_trained_model_and_scaler(model_path, scaler_path):
    with CustomObjectScope({'mse': MeanSquaredError()}):
        model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_new_city_data(new_city_data, scaler):
    weather_data = new_city_data.drop(columns=['datetime'])
    scaled_data = scaler.transform(weather_data)
    return scaled_data

def predict_next_72_hours(model, new_city_data, scaler, seq_length=24, pred_length=6, total_hours=72):
    new_city_data_normalized = preprocess_new_city_data(new_city_data, scaler)
    
    predictions_list = []
    current_data = new_city_data_normalized.copy()
    
    for _ in range(total_hours // pred_length):
        last_sequence = current_data[-seq_length:]
        last_sequence = np.expand_dims(last_sequence, axis=0)

        pred = model.predict(last_sequence)
        pred = pred.reshape(-1, pred.shape[2])
        pred_original_scale = scaler.inverse_transform(pred)

        predictions_list.extend(pred_original_scale)
        
        current_data = np.vstack([current_data, pred])[-seq_length:]
    
    return np.array(predictions_list)

# Load model and scaler at initialization
try:
    base_dir = os.path.dirname(__file__)  # Get the directory of the current file
    model_path = os.path.join(base_dir, 'weather_forecast_model.h5')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    
    model, scaler = load_trained_model_and_scaler(model_path, scaler_path)
    model_loaded = True
except FileNotFoundError as e:
    print(f"Model file not found: {str(e)}. Please ensure 'weather_forecast_model.h5' and 'scaler.pkl' are in the correct directory.")
    model_loaded = False
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_loaded = False

# API Endpoint to fetch AQI data
@app.route('/fetch-aqi', methods=['POST'])
def fetch_aqi():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    result = fetch_aqi_data(query)
    return jsonify(result)


# API Endpoint to fetch weather data
@app.route('/fetch-weather', methods=['POST'])
def fetch_weather():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400

    result = fetch_weather_data(query)
    return jsonify(result)


# New API Endpoint for weather predictions
@app.route('/predict-weather', methods=['POST'])
def predict_weather():
    if not model_loaded:
        return jsonify({"error": "Model not loaded properly"}), 500
    
    data = request.json
    city_name = data.get("city", "")
    if not city_name:
        return jsonify({"error": "City name is required"}), 400
    
    # API keys
    openweather_api_key = "613fd230fb49000cf12c9f6d60972070"
    visualcrossing_api_key = "HDV3C6DJY2PLXHLJPLLNT4Z8T"
    
    # Get city coordinates
    latitude, longitude = get_city_coordinates(city_name, openweather_api_key)
    if not latitude or not longitude:
        return jsonify({"error": "Could not find coordinates for the city"}), 404
    
    # Get past 24 hours of weather data
    now = datetime.now()
    end_time = now.strftime('%Y-%m-%d')
    start_time = (now - timedelta(hours=24)).strftime('%Y-%m-%d')
    
    new_city_data = fetch_visual_crossing_data(latitude, longitude, start_time, end_time, visualcrossing_api_key)
    if new_city_data is None:
        return jsonify({"error": "Failed to fetch historical weather data"}), 500
    
    if new_city_data.isnull().values.any():
        return jsonify({"error": "Incomplete weather data"}), 500
    
    # Process data and make predictions
    new_city_data['datetime'] = pd.to_datetime(new_city_data['datetime'])
    processed_new_city_data = new_city_data.tail(24)
    
    try:
        predictions = predict_next_72_hours(model, processed_new_city_data, scaler, seq_length=24, pred_length=6, total_hours=72)
        
        # Format predictions
        prediction_columns = ['temperature', 'humidity', 'pressure', 'cloud_cover', 'weather_code', 'wind_speed']
        predictions_df = pd.DataFrame(predictions, columns=prediction_columns)
        
        # Add datetime to predictions
        start_prediction_time = now
        prediction_times = [start_prediction_time + timedelta(hours=i) for i in range(1, 73)]
        predictions_df.insert(0, 'datetime', prediction_times)
        
        # Convert datetime to string for JSON serialization
        predictions_df['datetime'] = predictions_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            "city": city_name,
            "latitude": latitude,
            "longitude": longitude,
            "predictions": predictions_df.to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
