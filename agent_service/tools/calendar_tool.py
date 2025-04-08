# tools/calendar_tool.py
from langchain_core.tools import tool
import csv
from datetime import datetime

class CalendarTool:
    @staticmethod
    @tool
    def calendar_tool(date: str) -> str:
        """
        Check calendar for meetings on a specific date.

        Args:
            date_str: The date in YYYY-MM-DD format.

        Returns:
            The day of the week.
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        file_path = "calendar.csv"
        meetings = []

        try:
            with open(file_path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                meetings = [row['meeting'] for row in reader if row['date'] == date]
        except Exception:
            return f"No calendar file or error reading it for {date}."

        if meetings:
            meeting_list = ", ".join(meetings)
            return f"You have the following meetings on {date}: {meeting_list}."
        else:
            return f"You have no meetings scheduled on {date}."
