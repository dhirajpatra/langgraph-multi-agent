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
    def weather_tool(location: str, unit: str = "celsius") -> str:
        """
        Get the current weather for a given location.

        Args:
            location: The city and country to get the weather for.

        Returns:
            A dictionary containing the current weather information.
        """
        return (
            f"Location: {location}\n"
            f"Unit: {unit}\n"
            f"Weather: {WeatherTool.get_weather(location, unit)}"
        )

    @staticmethod
    def get_weather(city: str, unit: str) -> str:
        try:
            # url = f"https://weather.indianapi.in/india/weather?city={quote(city)}"
            # headers = {
            #     "x-api-key": os.getenv("WEATHER_API_KEY"),
            # }
            # response = requests.get(url, headers=headers)
            # if response.status_code == 200:
            #     data = response.json()
            #     today_forecast = data["weather"]["forecast"][0]
            #     max_temp = today_forecast["max_temp"]
            #     min_temp = today_forecast["min_temp"]
            #     description = today_forecast["description"]

            #     weather_summary = (
            #         f"The current weather in Bengaluru, India is {description} "
            #         f"with a max temperature of {max_temp}°C and min of {min_temp}°C."
            #     )
            #     return weather_summary
            # else:
            #     return f"Failed to fetch weather data. Status code: {response.status_code}"
            # Best Practice: Log tool execution for easier debugging
            logging.info(f"--- Tool: get_weather called for city: {city} ---")
            city_normalized = city.lower().replace(" ", "") # Basic input normalization

            # Mock weather data for simplicity (matching Step 1 structure)
            mock_weather_db = {
                    "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
                    "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
                    "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
                    "chicago": {"status": "success", "report": "The weather in Chicago is sunny with a temperature of 25°C."},
                    "toronto": {"status": "success", "report": "It's partly cloudy in Toronto with a temperature of 30°C."},
                    "chennai": {"status": "success", "report": "It's rainy in Chennai with a temperature of 35°C."},
                    "bengaluru": {"status": "success", "report": "It's sunny in Bengaluru with a temperature of 15°C."},
                    "new delhi": {"status": "success", "report": "It's cloudy in New Delhi with a temperature of 45°C."},
                    "kolkata": {"status": "success", "report": "It's sunny in Kolkata with a temperature of 35°C."},
                    "mumbai": {"status": "success", "report": "It's cloudy in Mumbai with a temperature of 30°C."},
            }

            # Best Practice: Handle potential errors gracefully within the tool
            if city_normalized in mock_weather_db:
                logging.info(f"weather agent Weather data for {city} found in mock database.")
                return mock_weather_db[city_normalized]
            else:
                return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}
        except Exception as e:
            return f"An error occurred while fetching weather data: {str(e)}"
