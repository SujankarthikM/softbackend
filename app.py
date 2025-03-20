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

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)
CORS(app)

# API keys and constants
OPENWEATHER_API_KEY = "613fd230fb49000cf12c9f6d60972070"  # For current weather, forecast, UV, historical
API_KEY2 = "7eeba6a8fe29e05163a9d8011bffcba3"             # For ranking (using a different key)
SCRAPINGDOG_API_KEY = "67dc04c3f7277221019e339b"
SCRAPINGDOG_URL = "https://api.scrapingdog.com/google"
VISUALCROSSING_API_KEY = "PSCJWRPMCEDJ3XNB9HQ5VUTJK"
DEFAULT_CITIES = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore",
                  "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]

# -------------------------------
# Utility Functions
# -------------------------------

# 1. Fetch AQI Data using Scrapingdog
def fetch_aqi_data(query):
    params = {
        "api_key": SCRAPINGDOG_API_KEY,
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

    aqi_url = links[0]
    response = requests.get(aqi_url)
    if response.status_code != 200:
        return {"error": "Failed to fetch AQI page"}, 500

    raw_html = response.text
    cleaned_html = ''
    for i in range(len(raw_html)):
        cleaned_html += raw_html[i]
        if raw_html[i:i + 41] == "rank-comparison px-body w-auto sm:mx-body":
            break
    cleaned_html = cleaned_html[:-13] + '</body></html>'

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

# 2. Fetch Historical Weather Data using Scrapingdog (from timeanddate.com)
def fetch_weather_data_history(query):
    params = {
        "api_key": SCRAPINGDOG_API_KEY,
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
    if 'organic_results' in data:
        links.extend(
            item['link']
            for item in data['organic_results']
            if 'link' in item and 'timeanddate.com' in item['link']
        )
    if not links:
        return {"error": "No valid weather links found"}, 404

    weather_url = links[0]
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(weather_url, headers=headers)
    if response.status_code != 200:
        return {"error": "Failed to fetch weather page"}, 500

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

    df = pd.DataFrame(weather_data)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df = df[['date', 'ts', 'ds', 'desc', 'temp', 'templow', 'baro', 'wind', 'wd', 'hum']]
    return df.to_dict(orient='records')

# 3. Fetch Current Weather Data using OpenWeatherMap (for /api/weather)
def fetch_weather_data_current(city):
    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params)
        if response.status_code == 200:
            data = response.json()
            weather_info = {
                "city": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "pressure": data["main"]["pressure"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
                "wind_direction": data["wind"].get("deg", 0),
                "cloud_cover": data["clouds"]["all"],
                "precipitation": data.get("rain", {}).get("1h", 0),
                "visibility": data.get("visibility", 0) / 1000,
                "weather": data["weather"][0]["description"].capitalize(),
                "weather_id": data["weather"][0]["id"],
                "weather_icon": data["weather"][0]["icon"],
                "lat": data["coord"]["lat"],
                "lon": data["coord"]["lon"],
                "timestamp": data["dt"]
            }
            return weather_info
        else:
            return {"error": f"Failed to retrieve data: {response.json().get('message', 'Unknown error')}"}
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

# 4. Fetch UV Index using OpenWeatherMap
def fetch_uv_index(lat, lon):
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    try:
        response = requests.get(UV_URL, params=params)
        if response.status_code == 200:
            return response.json().get("value", 0)
        return 0
    except:
        return 0

# 5. Fetch Forecast Data using OpenWeatherMap
def fetch_forecast_data(city):
    params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    try:
        response = requests.get(FORECAST_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            forecast_list = []
            for item in data.get("list", []):
                forecast_info = {
                    "timestamp": item["dt"],
                    "temperature": item["main"]["temp"],
                    "pressure": item["main"]["pressure"],
                    "humidity": item["main"]["humidity"],
                    "wind_speed": item["wind"]["speed"],
                    "wind_direction": item["wind"]["deg"],
                    "cloud_cover": item["clouds"]["all"],
                    "precipitation": item.get("rain", {}).get("3h", 0),
                    "weather": item["weather"][0]["description"].capitalize(),
                    "weather_id": item["weather"][0]["id"],
                    "weather_icon": item["weather"][0]["icon"]
                }
                forecast_list.append(forecast_info)
            return forecast_list
        else:
            return []
    except Exception as e:
        return []

# 6. Fetch Weather for Ranking (using API_KEY2)
def fetch_weather_for_ranking(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY2}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "city": city,
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data.get("wind", {}).get("speed", 0),
            "cloudiness": data.get("clouds", {}).get("all", 0),
            "precipitation": data.get("rain", {}).get("1h", 0)
        }
    return None

# 7. Rank Cities based on weather parameters
def rank_cities(cities):
    city_data = [fetch_weather_for_ranking(city) for city in cities]
    city_data = [c for c in city_data if c]
    if not city_data:
        return []
    weights = {"temperature": 0.25, "pressure": 0.10, "precipitation": 0.25,
               "humidity": 0.15, "wind_speed": 0.15, "cloudiness": 0.10}
    # For temperature and pressure, higher is better; for others, lower is better
    for key in ["temperature", "pressure"]:
        values = [c[key] for c in city_data]
        min_val, max_val = min(values), max(values)
        for c in city_data:
            c[f"norm_{key}"] = (c[key] - min_val) / (max_val - min_val) if max_val > min_val else 1
    for key in ["precipitation", "humidity", "wind_speed", "cloudiness"]:
        values = [c[key] for c in city_data]
        min_val, max_val = min(values), max(values)
        for c in city_data:
            c[f"norm_{key}"] = (max_val - c[key]) / (max_val - min_val) if max_val > min_val else 1
    for c in city_data:
        c["score"] = sum(weights[k] * c[f"norm_{k}"] for k in weights.keys())
    city_data.sort(key=lambda x: x["score"], reverse=True)
    return [{"rank": idx + 1, **c} for idx, c in enumerate(city_data)]

# 8. Functions for Weather Prediction
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
        "elements": "datetime,temp,humidity,pressure,cloudcover,conditions,windspeed"
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

# Load prediction model and scaler at initialization
try:
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'weather_forecast_model.h5')
    scaler_path = os.path.join(base_dir, 'scaler.pkl')
    model, scaler = load_trained_model_and_scaler(model_path, scaler_path)
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_loaded = False

# -------------------------------
# API Endpoints
# -------------------------------

# Endpoint 1: Fetch AQI
@app.route('/fetch-aqi', methods=['POST'])
def fetch_aqi():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    result = fetch_aqi_data(query)
    return jsonify(result)

# Endpoint 2: Fetch Historical Weather Data
@app.route('/fetch-weather', methods=['POST'])
def fetch_weather_history():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    result = fetch_weather_data_history(query)
    return jsonify(result)

# Endpoint 3: Weather Prediction (next 72 hours)
@app.route('/predict-weather', methods=['POST'])
def predict_weather():
    if not model_loaded:
        return jsonify({"error": "Model not loaded properly"}), 500
    data = request.json
    city_name = data.get("city", "")
    if not city_name:
        return jsonify({"error": "City name is required"}), 400
    latitude, longitude = get_city_coordinates(city_name, OPENWEATHER_API_KEY)
    if not latitude or not longitude:
        return jsonify({"error": "Could not find coordinates for the city"}), 404
    now = datetime.now()
    end_time = now.strftime('%Y-%m-%d')
    start_time = (now - timedelta(hours=24)).strftime('%Y-%m-%d')
    new_city_data = fetch_visual_crossing_data(latitude, longitude, start_time, end_time, VISUALCROSSING_API_KEY)
    if new_city_data is None:
        return jsonify({"error": "Failed to fetch historical weather data"}), 500
    if new_city_data.isnull().values.any():
        return jsonify({"error": "Incomplete weather data"}), 500
    new_city_data['datetime'] = pd.to_datetime(new_city_data['datetime'])
    processed_new_city_data = new_city_data.tail(24)
    try:
        predictions = predict_next_72_hours(model, processed_new_city_data, scaler, seq_length=24, pred_length=6, total_hours=72)
        prediction_columns = ['temperature', 'humidity', 'pressure', 'cloud_cover', 'weather_code', 'wind_speed']
        predictions_df = pd.DataFrame(predictions, columns=prediction_columns)
        start_prediction_time = now
        prediction_times = [start_prediction_time + timedelta(hours=i) for i in range(1, 73)]
        predictions_df.insert(0, 'datetime', prediction_times)
        predictions_df['datetime'] = predictions_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        return jsonify({
            "city": city_name,
            "latitude": latitude,
            "longitude": longitude,
            "predictions": predictions_df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

# Endpoint 4: City Rankings (Default and Custom)
@app.route('/rankings/default', methods=['GET'])
def get_default_rankings():
    rankings = rank_cities(DEFAULT_CITIES)
    return jsonify(rankings)

@app.route('/rankings/custom', methods=['POST'])
def get_custom_rankings():
    data = request.json
    selected_cities = data.get("cities", [])
    if not selected_cities:
        return jsonify({"error": "No cities provided"}), 400
    rankings = rank_cities(selected_cities)
    return jsonify(rankings)

# Endpoint 5: Current Weather, UV Index, and Forecast
@app.route('/api/weather', methods=['POST'])
def get_current_weather():
    data = request.json
    city = data.get('city')
    if not city:
        return jsonify({"error": "City name is required"}), 400
    current_weather = fetch_weather_data_current(city)
    if "error" in current_weather:
        return jsonify(current_weather), 404
    lat, lon = current_weather.get("lat", 0), current_weather.get("lon", 0)
    current_weather["uv_index"] = fetch_uv_index(lat, lon)
    forecast = fetch_forecast_data(city)
    return jsonify({
        "current": current_weather,
        "forecast": forecast
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
