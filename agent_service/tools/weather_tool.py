# tools/weather_tool.py
from langchain_core.tools import tool
import logging
from dotenv import load_dotenv
from tools.ipinfo import get_current_location

logging.basicConfig(level=logging.INFO)
load_dotenv()

@tool("weather_tool", description="Get the current weather for a given location.")
def weather_tool(location: str, unit: str = "celsius") -> dict:
    """
    Get the current weather for a given location.

    Args:
        location: The city and country to get the weather for.
        unit: The unit of temperature (default is Celsius).

    Returns:
        A dictionary containing the current weather information.
    """
    try:    
        ip_info = get_current_location()
        location = location or ip_info.get("city")
        weather_data = get_weather(location, unit)
        return {"status": "success", "report": weather_data}
    except Exception as e:
        logging.error(f"Error in weather_tool: {e}")
        return {"status": "error", "error_message": f"Error fetching weather: {str(e)}"}

def get_weather(city: str, unit: str) -> str:
    """
    Fetch the weather information for the specified city.
    Here, we simulate the weather data for simplicity.
    """
    logging.info(f"--- Tool: get_weather called for city: {city} ---")
    city_normalized = city.lower().replace(" ", "")  # Normalize the input
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
            # Mock weather data for simplicity (in place of real API calls)
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

    # Return the weather info if city exists in mock data
    if city_normalized in mock_weather_db:
        logging.info(f"Weather data for {city} found in mock database.")
        return mock_weather_db[city_normalized]
    else:
        logging.warning(f"No weather information for {city}.")
        return f"Sorry, I don't have weather information for '{city}'."
