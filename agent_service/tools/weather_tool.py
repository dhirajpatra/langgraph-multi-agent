# tools/weather_tool.py
from langchain_core.tools import tool
import requests
from urllib.parse import quote
import os
from dotenv import load_dotenv
import logging
logging.basicConfig(level=logging.INFO)
load_dotenv()

class WeatherTool:
    @staticmethod
    @tool
    def weather_tool(location: str, unit: str = "celsius") -> dict:
        """
        Get the current weather for a given location.

        Args:
            location: The city and country to get the weather for.

        Returns:
            A dictionary containing the current weather information.
        """
        return {
            "location": location,
            "unit": unit,
            "weather": WeatherTool.get_weather(location),
        }
    
    @staticmethod
    def get_weather(city: str) -> str:
        try:
            url = f"https://weather.indianapi.in/india/weather?city={quote(city)}"
            headers = {
                "x-api-key": os.getenv("WEATHER_API_KEY"),
            }
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                today_forecast = data["weather"]["forecast"][0]
                max_temp = today_forecast["max_temp"]
                min_temp = today_forecast["min_temp"]
                description = today_forecast["description"]

                weather_summary = (
                    f"The current weather in Bengaluru, India is {description} "
                    f"with a max temperature of {max_temp}°C and min of {min_temp}°C."
                )
                return weather_summary
            else:
                return f"Failed to fetch weather data. Status code: {response.status_code}"
        except Exception as e:
            return f"An error occurred while fetching weather data: {str(e)}"
