import requests
from urllib.parse import quote


def get_weather(city: str) -> str:
    """
    通过wttr.in查询实时天气信息。
    """
    city_encoded = quote(city)
    url = f"https://wttr.in/{city_encoded}?format=j1"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        current_condition = data["current_condition"][0]
        weather_desc = current_condition["weatherDesc"][0]["value"]
        temp_c = current_condition["temp_C"]

        return f"{city} current weather: {weather_desc}, temperature {temp_c} degrees Celsius"

    except requests.exceptions.RequestException as e:
        return f"Error: Network problem encountered when querying weather - {e}"
    except (KeyError, IndexError) as e:
        return f"Error: Failed to parse weather data, city name may be invalid - {e}"
