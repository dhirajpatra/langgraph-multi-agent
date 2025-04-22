# tools/calendar_tool.py

from langchain_core.tools import tool
from pydantic import BaseModel
import csv
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)


class CalendarToolArgs(BaseModel):
    pass  # No arguments needed


@tool(args_schema=CalendarToolArgs, description="Check calendar for meetings on today's date.")
def calendar_tool(tool_call_id: str | None = None) -> dict:
    """
    Check calendar for meetings on today's date.
    """
    today = datetime.today().strftime("%Y-%m-%d")
    file_path = "calendar.csv"

    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            meetings = [row['meeting'] for row in reader if row.get('date') == today]
    except Exception as e:
        logging.error(f"[calendar_tool] Error reading calendar: {e}")
        return {"status": "error", "error_message": f"Error reading calendar: {str(e)}"}

    if meetings:
        logging.info(f"[calendar_tool] Meetings on {today}: {meetings}")
        return {"status": "success", "report": f"Meetings on {today}: {meetings}"}
    else:
        logging.warning(f"[calendar_tool] No meetings found for {today}.")
        return {"status": "error", "error_message": f"No meetings scheduled for {today}."}
