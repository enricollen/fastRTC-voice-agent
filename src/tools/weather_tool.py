"""
tools implementation
"""
import os
import requests
from loguru import logger
from llama_index.core.tools import FunctionTool


def get_weather(location: str, unit: str = "celsius") -> str:
    """
    get the current weather in a given location using meteosource api.
    
    args:
        location (str): the city and state, e.g. san francisco, ca or place_id like 'london', 'rome'
        unit (str): the unit of temperature, either celsius or fahrenheit
        
    returns:
        str: a string describing the current weather
    """
    try:
        logger.info(f"getting weather for {location} in {unit}")
        
        # get api key from environment variable
        api_key = os.getenv("METEO_API")
        
        if not api_key:
            logger.error("meteo_api environment variable not set")
            return "error: weather api key not configured."
        
        # normalize location to place_id format (lowercase, replace spaces with underscores)
        place_id = location.lower().replace(" ", "_").replace(",", "")
        
        # build the api endpoint url
        url = (
            f"https://www.meteosource.com/api/v1/free/point"
            f"?place_id={place_id}"
            f"&sections=current"
            f"&timezone=UTC"
            f"&language=en"
            f"&units=metric"
            f"&key={api_key}"
        )
        
        # make the api call
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # extract current weather data
        if "current" not in data:
            logger.warning(f"no current weather data found for {location}")
            return f"no weather data found for {location}."
        
        current = data["current"]
        
        # extract weather information
        temp_celsius = current.get("temperature", 25)
        summary = current.get("summary", "n/a")
        wind_speed = current.get("wind", {}).get("speed", 0)
        cloud_cover = current.get("cloud_cover", 0)
        precipitation = current.get("precipitation", {}).get("total", 0)
        
        # convert temperature if needed
        if unit == "fahrenheit":
            temp = (temp_celsius * 9/5) + 32
            temp_unit = "°F"
        else:
            temp = temp_celsius
            temp_unit = "°C"
        
        # format response in english
        response_text = f"in {location} the weather is {summary}, temperature {temp:.0f}{temp_unit}"
        
        # add wind info if significant
        if wind_speed > 3:
            response_text += f", wind {wind_speed:.0f} meters per second"
        
        # add precipitation info if present
        if precipitation > 0:
            response_text += f", precipitation {precipitation:.1f} millimeters"
        
        response_text += "."
        
        # add detailed log for debugging
        logger.debug(f"weather data from api: {current}")
        logger.debug(f"formatted response: {response_text}")
        
        return response_text
        
    except requests.exceptions.RequestException as e:
        logger.error(f"api request error: {str(e)}")
        return f"unable to retrieve weather information for {location}."
    except Exception as e:
        logger.error(f"error getting weather: {str(e)}")
        return f"error occurred while getting weather: {str(e)}"