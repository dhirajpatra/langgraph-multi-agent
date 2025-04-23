# tools/weather_tool.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging
from dotenv import load_dotenv
from tools.ipinfo import get_current_location
import os

logging.basicConfig(level=logging.INFO)
load_dotenv()

class WeatherToolArgs(BaseModel):
    location: str = Field(description="The city or country to get the weather for.")
    unit: str = Field(default="celsius", description="The unit of temperature (default is Celsius).")

@tool(args_schema=WeatherToolArgs, description="Get the current weather for a given location.")
def weather_tool(location: str, unit: str = "celsius", tool_call_id: str | None = None) -> dict:
    """
    Get the current weather for a given location.
    """
    try:    
        ip_info = get_current_location()
        logging.info(f"******************** [weather_tool] IP Info: {ip_info}")
        location = ip_info.get("city")
        logging.info(f"***************** [weather_tool] Location: {location}")
        weather_data = get_weather(location, unit)
        return {"status": "success", "report": weather_data}
    except Exception as e:
        logging.error(f"[weather_tool] Error: {e}")
        return {"status": "error", "error_message": f"Error fetching weather: {str(e)}"}

def get_weather(city: str, unit: str) -> str:
    """
    Fetch the weather information for the specified city.
    Simulated weather data returned if real API call is not used.
    """
    logging.info(f"[get_weather] Called for city: {city} (unit: {unit})")
    city_normalized = city.lower().replace(" ", "")

    # Base mock data in Celsius
    mock_weather_db_celsius = {
        "newyork": {"condition": "sunny", "temp": 25},
        "london": {"condition": "cloudy", "temp": 15},
        "tokyo": {"condition": "light rain", "temp": 18},
        "chicago": {"condition": "sunny", "temp": 25},
        "toronto": {"condition": "partly cloudy", "temp": 30},
        "chennai": {"condition": "rainy", "temp": 35},
        "bengaluru": {"condition": "sunny", "temp": 15},
        "newdelhi": {"condition": "cloudy", "temp": 45},
        "kolkata": {"condition": "sunny", "temp": 35},
        "mumbai": {"condition": "cloudy", "temp": 30},
    }

    if city_normalized in mock_weather_db_celsius:
        data = mock_weather_db_celsius[city_normalized]
        temp = data["temp"]
        
        # Convert temperature if needed
        if unit.lower() in ["fahrenheit", "f"]:
            temp = (temp * 9/5) + 32
            temp_unit = "°F"
        else:
            temp_unit = "°C"
        
        logging.info(f"[get_weather] Found mock weather data for: {city}")
        return f"The weather in {city.title()} is {data['condition']} with a temperature of {int(temp)}{temp_unit}."
    else:
        logging.warning(f"[get_weather] No mock weather data for: {city}")
        return f"Sorry, I don't have weather information for '{city}'."

    # Uncomment below to enable real API integration
    """
    try:
        url = f"https://weather.indianapi.in/india/weather?city={quote(city)}"
        headers = {"x-api-key": os.getenv("WEATHER_API_KEY")}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            forecast = data["weather"]["forecast"][0]
            return (
                f"The current weather in {city.title()}, India is {forecast['description']} "
                f"with a max temperature of {forecast['max_temp']}°C and min of {forecast['min_temp']}°C."
            )
        else:
            return f"Failed to fetch weather data. Status code: {response.status_code}"
    except Exception as api_error:
        logging.error(f"[get_weather] API error: {api_error}")
        return f"API error: {api_error}"
    """