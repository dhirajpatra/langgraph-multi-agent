# tools/calendar_tool.py
from langchain_core.tools import tool
import csv
from datetime import datetime

class CalendarTool:
    @staticmethod
    @tool
    def calendar_tool(date: str) -> dict:
        """
        Check calendar for meetings on a specific date.

        Args:
            date: The date in YYYY-MM-DD format.

        Returns:
            A dictionary with date, status, and meetings.
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        file_path = "calendar.csv"
        meetings = []

        try:
            with open(file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                meetings = [row['meeting'] for row in reader if row['date'] == date]
        except Exception as e:
            return {
                "date": date,
                "status": "error",
                "message": f"Error reading calendar: {str(e)}",
                "meetings": []
            }

        if meetings:
            return {
                "date": date,
                "status": "found",
                "meetings": meetings
            }
        else:
            return {
                "date": date,
                "status": "none",
                "meetings": []
            }
