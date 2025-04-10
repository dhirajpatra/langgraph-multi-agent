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
            logging.info(f"+++++++++++++++++++++++++++++++++++Reading calendar from {file_path} for date: {date}")
            with open(file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                meetings = [row['meeting'] for row in reader if row['date'] == date]
        except Exception as e:
            return f"Error reading calendar: {str(e)}"
        logging.info(f"+++++++++++++++++++++++++++++++++++Meetings found: {meetings}")
        if meetings:
            return f"Meetings on {date}: {', '.join(meetings)}"
        else:
            return f"No meetings found on {date}."
