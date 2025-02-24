from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# API key for scrapingdog
API_KEY = "67b81625286dcf801806cfa3"
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
    data_section = data_section.replace('true', 'True').replace('\/', '')

    try:
        weather_data = eval(data_section)['detail']
    except Exception as e:
        return {"error": f"Failed to parse weather data: {str(e)}"}, 500

    # Process weather data into DataFrame
    df = pd.DataFrame(weather_data)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df = df[['date', 'ts', 'ds', 'desc', 'temp', 'templow', 'baro', 'wind', 'wd', 'hum']]
    return df.to_dict(orient='records')


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


if __name__ == "__main__":
    app.run(debug=True)
    