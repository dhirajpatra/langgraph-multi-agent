# tools/calendar_tool.py
from langchain_core.tools import tool
import csv
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)

class CalendarTool:
    @staticmethod
    @tool
    def calendar_tool() -> str:
        """
        Check calendar for meetings on a specific date.

        Args:
            date: The date in YYYY-MM-DD format.

        Returns:
            A string with date, and meetings.
        """
        date = datetime.today().strftime("%Y-%m-%d")

        file_path = "calendar.csv"
        meetings = []

        try:
            with open(file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                meetings = [row['meeting'] for row in reader if row['date'] == date]
        except Exception as e:
            return f"Error reading calendar: {str(e)}"
        if meetings:
            logging.info(f"Calendar tool Meetings found on {date}: {meetings}")
            return {"status": "success", "report": f"Meetings found on {date}: {meetings}"}
        else:
            return {"status": "error", "error_message": f"Sorry, I don't have calendar information for '{date}'."}
