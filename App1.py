from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_KEY = "7eeba6a8fe29e05163a9d8011bffcba3"
DEFAULT_CITIES = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]

def fetch_weather(city):
    """Fetch weather data from OpenWeather API for a city."""
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = {
            "city": city,
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"] if "wind" in data else 0,
            "cloudiness": data["clouds"]["all"] if "clouds" in data else 0,
            "precipitation": data["rain"]["1h"] if "rain" in data and "1h" in data["rain"] else 0
        }
        return weather
    return None

def rank_cities(cities):
    """Fetch weather data for each city, normalize parameters, and calculate a weighted score."""
    city_data = [fetch_weather(city) for city in cities]
    city_data = [c for c in city_data if c]  # Remove cities that returned no data
    if not city_data:
        return []

    # Weights for each parameter
    weights = {
        "temperature": 0.25,   # higher temperature is better
        "pressure": 0.10,      # higher pressure is considered better
        "precipitation": 0.25, # lower precipitation is better
        "humidity": 0.15,      # lower humidity is better
        "wind_speed": 0.15,    # lower wind speed is better
        "cloudiness": 0.10,    # lower cloudiness is better
    }

    # Normalize positive parameters: temperature and pressure (higher is better)
    for key in ["temperature", "pressure"]:
        values = [c[key] for c in city_data]
        min_val, max_val = min(values), max(values)
        for c in city_data:
            c[f"norm_{key}"] = (c[key] - min_val) / (max_val - min_val) if max_val > min_val else 1

    # Normalize negative parameters: precipitation, humidity, wind_speed, cloudiness (lower is better)
    for key in ["precipitation", "humidity", "wind_speed", "cloudiness"]:
        values = [c[key] for c in city_data]
        min_val, max_val = min(values), max(values)
        for c in city_data:
            c[f"norm_{key}"] = (max_val - c[key]) / (max_val - min_val) if max_val > min_val else 1

    # Compute weighted score
    for c in city_data:
        score = 0
        for key, weight in weights.items():
            score += weight * c[f"norm_{key}"]
        c["score"] = score

    # Sort by score (highest first) and add ranking
    city_data.sort(key=lambda x: x["score"], reverse=True)
    return [{"rank": idx + 1, **city} for idx, city in enumerate(city_data)]

@app.route("/rankings/default", methods=["GET"])
def get_default_rankings():
    rankings = rank_cities(DEFAULT_CITIES)
    return jsonify(rankings)

@app.route("/rankings/custom", methods=["POST"])
def get_custom_rankings():
    data = request.json
    selected_cities = data.get("cities", [])
    if not selected_cities:
        return jsonify({"error": "No cities provided"}), 400
    rankings = rank_cities(selected_cities)
    return jsonify(rankings)

if __name__ == "__main__":
    app.run(debug=True,port=5002)
