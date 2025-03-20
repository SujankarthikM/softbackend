from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# OpenWeatherMap API Key
API_KEY = "613fd230fb49000cf12c9f6d60972070"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
UV_URL = "https://api.openweathermap.org/data/2.5/uvi"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"

@app.route('/api/weather', methods=['POST'])
def get_weather():
    data = request.json
    city = data.get('city')
    
    if not city:
        return jsonify({"error": "City name is required"}), 400
    
    # Fetch current weather data
    current_weather = fetch_weather_data(city)
    if "error" in current_weather:
        return jsonify(current_weather), 404
    
    # Fetch UV index using coordinates
    lat, lon = current_weather.get("lat", 0), current_weather.get("lon", 0)
    uv_index = fetch_uv_index(lat, lon)
    current_weather["uv_index"] = uv_index
    
    # Fetch forecast data
    forecast = fetch_forecast_data(city)
    
    return jsonify({
        "current": current_weather,
        "forecast": forecast
    })

def fetch_weather_data(city):
    """Fetches weather data for the given city"""
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
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
                "visibility": data.get("visibility", 0) / 1000,  # Convert to km
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

def fetch_uv_index(lat, lon):
    """Fetches UV index for given latitude and longitude"""
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY
    }
    
    try:
        response = requests.get(UV_URL, params=params)
        if response.status_code == 200:
            return response.json().get("value", 0)
        return 0
    except:
        return 0

def fetch_forecast_data(city):
    """Fetches 5-day forecast data for the given city"""
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }
    
    try:
        response = requests.get(FORECAST_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            forecast_list = []
            
            # Process each forecast time
            for item in data["list"]:
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

if __name__ == "__main__":
    app.run(debug=True,port=5001)
