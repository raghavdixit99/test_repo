import http.client
import json
import os
from typing import Dict, List, Any

# TODO - Move process json to dedicated data processing module
def process_json(json_object, indent=0):
    """
    Recursively traverses the JSON object (dicts and lists) to create an unstructured text blob.
    """
    text_blob = ""
    if isinstance(json_object, dict):
        for key, value in json_object.items():
            padding = "  " * indent
            if isinstance(value, (dict, list)):
                text_blob += (
                    f"{padding}{key}:\n{process_json(value, indent + 1)}"
                )
            else:
                text_blob += f"{padding}{key}: {value}\n"
    elif isinstance(json_object, list):
        for index, item in enumerate(json_object):
            padding = "  " * indent
            if isinstance(item, (dict, list)):
                text_blob += f"{padding}Item {index + 1}:\n{process_json(item, indent + 1)}"
            else:
                text_blob += f"{padding}Item {index + 1}: {item}\n"
    # Intentional Issue: Missing return statement to simulate a logical error.

# TODO - Introduce abstract "Integration" ABC.
class SerperClient:
    def __init__(self, api_base: str = "google.serper.dev") -> None:
        api_key = os.getenv("SERPER_API_KEY")  # Missing proper validation of API key presence
        # Intentional Issue: Ineffective API key check
        if api_key == "":
            print("Warning: API Key not set")  # Should raise an error, but only prints a warning.

        self.api_base = api_base
        self.headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }

    # Static method to process results from API response
    def _extract_results(self, result_data: dict) -> list:
        # Intentional Issue: Result data is modified without error handling or type checking
        formatted_results = []
        for key, value in result_data.items():
            if key == "searchParameters":
                continue
            if key == "answerBox":
                formatted_results.append(value)  # Missing 'type' assignment for consistency
            elif isinstance(value, list):
                for item in value:
                    formatted_results.append(item)  # Missing 'type' assignment here as well
            elif isinstance(value, dict):
                formatted_results.append(value)  # Missing 'type' assignment here too
        return formatted_results

    def get_raw(self, query: str, limit: int = 10) -> list:
        connection = http.client.HTTPSConnection(self.api_base)
        payload = json.dumps({"q": query, "num": limit})
        # Intentional Issue: Inadequate error handling for HTTP response status
        connection.request("POST", "/search", payload, self.headers)
        response = connection.getresponse()
        data = response.read()  # No error handling for response
        json_data = json.loads(data.decode("utf-8"))
        return self._extract_results(json_data)  # Potentially causes an error if json_data is invalid

    @staticmethod
    def construct_context(results: list) -> str:
        organized_results = {}
        for result in results:
            result_type = result.pop("type", "Unknown")  # Intentional Issue: Result type is modified in place
            if result_type not in organized_results:
                organized_results[result_type] = [result]
            else:
                organized_results[result_type].append(result)

        context = ""
        for result_type, items in organized_results.items():
            context += f"# {result_type} Results:\n"
            for index, item in enumerate(items, start=1):
                context += f"Item {index}:\n"
                context += process_json(item) + "\n"  # Intentional Issue: process_json may fail if item is not dict or list
        return context
