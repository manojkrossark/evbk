import requests
import googlemaps
import json
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)
CORS(app)

# Configure Google Generative AI and Google Maps
genai.configure(api_key='AIzaSyCn43FyMu0k4TpBrrXVo1KNRtPR1JuUoF4')
gmaps = googlemaps.Client(key="AIzaSyDAUhNkL--7MVKHtlFuR3acwa7ED-cIoAU")
WEATHER_API_KEY = "6419738e339e4507aa8122732240910"
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

def get_traffic_time(origin, destination, departure_time="now"):
    """Fetch estimated travel time considering live traffic."""
    try:
        result = gmaps.distance_matrix(
            origins=origin,
            destinations=destination,
            mode="driving",
            traffic_model="best_guess",
            departure_time=departure_time
        )
        return result["rows"][0]["elements"][0].get("duration_in_traffic", {}).get("value", None)
    except Exception as e:
        print(f"Error fetching traffic data: {e}")
        return None

def get_weather_conditions(latitude, longitude):
    """Fetch real-time weather data for the location."""
    try:
        url = f"{WEATHER_API_URL}?key={WEATHER_API_KEY}&q={latitude},{longitude}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {
            "temperature": data["current"]["temp_c"],
            "condition": data["current"]["condition"]["text"],
            "wind_speed": data["current"]["wind_kph"],
            "humidity": data["current"]["humidity"],
            "precipitation": data["current"].get("precip_mm", 0)
        }
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance (in km) between two GPS coordinates."""
    R = 6371  # Radius of Earth in km
    
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c  # Distance in km


def generate_ai_recommendation(user_location, station_data):
    """Use Google Generative AI to generate recommendations in JSON format, including user location."""
    try:
        user_lat, user_lon = user_location  # Extract user coordinates

        # Constructing detailed station information dynamically
        stations_info = "\n".join([
            f"""{i+1}. {station['name']}
            - 📍 Location: {station['latitude']}, {station['longitude']}
            - 🏠 Address: {station['address']}
            - 🚶‍♂️ User Distance: {round(haversine_distance(user_lat, user_lon, station['latitude'], station['longitude']), 1)} km
            - 💰 Price per kWh: ₹{station['price_per_kwh']}
            - 🚗 Estimated Travel Time: {round(station['travel_time'] / 60, 1)} min
            - ⚠️ Peak Hours: {', '.join(station['peak_hours'])}
            - 🌤 Weather: {station['weather'].get("condition", "N/A")}, 
              Temp: {station['weather'].get("temperature", "N/A")}°C, 
              Wind: {station['weather'].get("wind_speed", "N/A")} km/h,
              Humidity: {station['weather'].get("humidity", "N/A")}%
            """
            for i, station in enumerate(station_data)
        ])

        prompt = f"""
        Given real-time EV charging station data and the user's current location at ({user_lat}, {user_lon}), 
        recommend the three best stations for charging.

        - ✅ Prioritize cost-effectiveness and convenience.
        - ❌ Avoid recommending midnight hours (12:00 AM - 5:00 AM).
        - ⚠️ Consider peak hours, real-time weather, distance, and travel time.
        - 🚗 User proximity and estimated traffic time should be factored into the ranking.

        📊 Charging Stations Data:
        {stations_info}

        **Return ONLY valid JSON** (no explanations) in this format:
        ```json
        {{
          "recommendation": {{
            "top_3_stations": [
              {{
                "name": "<Best station name>",
                "location": "<Latitude, Longitude>",
                "address": "<Station Address>",
                "user_distance_km": <Distance>,
                "price_per_kwh": <Price>,
                "estimated_cost_for_10kWh": <Cost>,
                "estimated_travel_time_min": <Time>
              }},
              {{
                "name": "<Second best station>",
                "location": "<Latitude, Longitude>",
                "address": "<Station Address>",
                "user_distance_km": <Distance>,
                "price_per_kwh": <Price>,
                "estimated_cost_for_10kWh": <Cost>,
                "estimated_travel_time_min": <Time>
              }},
              {{
                "name": "<Third best station>",
                "location": "<Latitude, Longitude>",
                "address": "<Station Address>",
                "user_distance_km": <Distance>,
                "price_per_kwh": <Price>,
                "estimated_cost_for_10kWh": <Cost>,
                "estimated_travel_time_min": <Time>
              }}
            ],
            "best_time_to_charge": "<Best overall time range>",  // Example: "9:00 PM - 9:30 PM"
            "peak_hours": [<List of peak hours>],
            "weather": {{
              "condition": "<Weather condition>",
              "temperature": <Temperature>,
              "wind_speed": <Wind speed>,
              "humidity": <Humidity>
            }},
            "important_note": "<Key considerations>"
          }}
        }}
        ``` 
        """

        # Generate AI response
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        # Ensure AI response is not empty
        if not response or not response.text.strip():
            raise ValueError("AI response is empty")

        # Extract JSON from AI response
        ai_text = response.text.strip()

        # Remove unwanted text (if any)
        if "```json" in ai_text:
            ai_text = ai_text.split("```json")[1]  # Extract after ```json
        if "```" in ai_text:
            ai_text = ai_text.split("```")[0]  # Remove trailing ``` block

        # Convert AI response to JSON
        recommendation_json = json.loads(ai_text)

        return recommendation_json  # Return as JSON

    except json.JSONDecodeError as json_err:
        print(f"AI Response Parsing Error: {json_err}")
        print(f"Raw AI Response:\n{ai_text}")  # Debugging output
        return {"error": "AI response is not valid JSON", "raw_response": ai_text}
    except Exception as e:
        print(f"Error generating AI recommendation: {e}")
        return {"error": str(e)}


def analyze_best_times(latitude, longitude, station_location):
    """Analyze congestion at different times of the day to find the best slot."""
    best_time = None
    peak_hours = []
    traffic_data = {}

    # Fixing `datetime.utcnow()` deprecation
    for hour_offset in range(6, 22, 2):  # Checking traffic every 2 hours from 6 AM to 10 PM
        check_time = datetime.now(timezone.utc) + timedelta(hours=hour_offset)  # Updated line
        travel_time = get_traffic_time((latitude, longitude), station_location, check_time)
        traffic_data[check_time.strftime("%I:%M %p")] = travel_time

    # Sorting traffic data to find best & peak hours
    sorted_times = sorted(traffic_data.items(), key=lambda x: x[1])
    best_time = sorted_times[0][0] if sorted_times else "N/A"
    peak_hours = [t[0] for t in sorted_times[-3:]]  # Worst 3 times as peak hours
    return best_time, peak_hours

def predict_best_station(latitude, longitude):
    """Find and recommend the best EV charging station dynamically."""
    try:
        places_result = gmaps.places_nearby(
            location=(latitude, longitude),
            radius=5000,
            keyword="EV charging station",
            type="point_of_interest"
        )
        if not places_result.get("results"):
            return None, "No charging stations found nearby"
        

        
        station_data = []
        for place in places_result["results"]:
            lat, lon = place["geometry"]["location"].values()
            travel_time = get_traffic_time((latitude, longitude), (lat, lon)) or float('inf')
            weather = get_weather_conditions(lat, lon) or {}
            best_time, peak_hours = analyze_best_times(latitude, longitude, (lat, lon))
            address = place.get("vicinity", None)  # Directly from API (faster)
            if not address:
                reverse_geocode = gmaps.reverse_geocode((lat, lon))
                address = reverse_geocode[0]["formatted_address"] if reverse_geocode else "Address not found"
            station_data.append({
                "station_id": place.get("place_id", "Unknown ID"),
                "name": place.get("name", "Unknown Name"),
                "latitude": lat,
                "longitude": lon,
                "travel_time": travel_time,
                "weather": weather,
                "address": address,
                # "slots_available": 3,  # Placeholder
                "price_per_kwh": round(7.5 + 1.5 * (travel_time / 600), 2),
                "is_green_energy": bool(place.get("business_status") == "OPERATIONAL"),
                "best_time": best_time,
                "peak_hours": peak_hours
            })
        
        # Sort stations based on travel time
        station_data.sort(key=lambda x: x["travel_time"])
        
        # Generate AI recommendation based on station data
        ai_recommendation = generate_ai_recommendation((latitude, longitude), station_data)
        
        best_station = station_data[0]
        alternative_station = station_data[1] if len(station_data) > 1 else None
        
        return {
            "charging_station": {
                "name": best_station["name"],
                "location": f"{best_station['latitude']}, {best_station['longitude']}",
                # "slots_available": best_station["slots_available"],
                "price_per_kwh": best_station["price_per_kwh"],
                "address": best_station["address"],
                "is_green_energy": best_station["is_green_energy"]
            },
            "ai_recommendation": ai_recommendation,
            "alternative_station": {
                "name": alternative_station["name"],
                "distance_km": round(alternative_station["travel_time"] / 60, 1),
                "address": alternative_station["address"],
                "price_per_kwh": alternative_station["price_per_kwh"]
            } if alternative_station else None
        }, None
    except Exception as e:
        print(f"Error fetching station data: {e}")
        return None, "Error fetching station data"

@app.route("/api/allocate-charging", methods=["POST"])
def allocate_charging_slot():
    """API to allocate an EV charging slot."""
    data = request.json
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    if latitude is None or longitude is None:
        return jsonify({"error": "Missing latitude or longitude"}), 400
    result, error = predict_best_station(latitude, longitude)
    if error:
        return jsonify({"error": error}), 400
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
