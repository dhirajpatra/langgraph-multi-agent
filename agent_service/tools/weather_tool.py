# tools/weather_tool.py
from langchain_core.tools import tool

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
            "weather": f"28° {unit} and sunny"
        }
