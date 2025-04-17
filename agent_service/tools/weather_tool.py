# tools/weather_tool.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import logging
from dotenv import load_dotenv
import os
# import requests
# from urllib.parse import quote

logging.basicConfig(level=logging.INFO)
load_dotenv()


class WeatherToolArgs(BaseModel):
    location: str = Field(description="The city and country to get the weather for.")
    unit: str = Field(default="celsius", description="The unit of temperature (default is Celsius).")


@tool(args_schema=WeatherToolArgs, description="Get the current weather for a given location.")
def weather_tool(location: str, unit: str = "celsius", tool_call_id: str | None = None) -> dict:
    """
    Get the current weather for a given location.
    """
    try:
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

    # Mock data for development/testing
    mock_weather_db = {
        "newyork": "The weather in New York is sunny with a temperature of 25°C.",
        "london": "It's cloudy in London with a temperature of 15°C.",
        "tokyo": "Tokyo is experiencing light rain and a temperature of 18°C.",
        "chicago": "The weather in Chicago is sunny with a temperature of 25°C.",
        "toronto": "It's partly cloudy in Toronto with a temperature of 30°C.",
        "chennai": "It's rainy in Chennai with a temperature of 35°C.",
        "bengaluru": "It's sunny in Bengaluru with a temperature of 15°C.",
        "newdelhi": "It's cloudy in New Delhi with a temperature of 45°C.",
        "kolkata": "It's sunny in Kolkata with a temperature of 35°C.",
        "mumbai": "It's cloudy in Mumbai with a temperature of 30°C.",
    }

    if city_normalized in mock_weather_db:
        logging.info(f"[get_weather] Found mock weather data for: {city}")
        return mock_weather_db[city_normalized]
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
