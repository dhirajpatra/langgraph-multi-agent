# tools/calendar_tool.py
from langchain_core.tools import tool
import csv
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)

@tool("check_calendar_today")
def calendar_tool() -> dict:
    """
    Check calendar for meetings on today's date.

    Returns:
        A dictionary with status and report or error message.
    """
    date = datetime.today().strftime("%Y-%m-%d")
    file_path = "calendar.csv"
    meetings = []

    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            meetings = [row['meeting'] for row in reader if row['date'] == date]
    except Exception as e:
        logging.error(f"Error reading calendar: {e}")
        return {
            "status": "error",
            "error_message": f"Error reading calendar: {str(e)}"
        }

    if meetings:
        logging.info(f"Meetings found on {date}: {meetings}")
        return {
            "status": "success",
            "report": f"Meetings found on {date}: {meetings}"
        }
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I don't have calendar information for '{date}'."
        }
