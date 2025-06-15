"""
tools implementation
"""
from loguru import logger
from llama_index.core.tools import FunctionTool


def get_weather(location: str, unit: str = "celsius") -> str:
    """
    get the current weather in a given location.
    
    args:
        location (str): the city and state, e.g. san francisco, ca
        unit (str): the unit of temperature, either celsius or fahrenheit
        
    returns:
        str: a string describing the current weather
    """
    try:
        logger.info(f"getting weather for {location} in {unit}")
        
        # mock data
        weather_conditions = {
            "rome": {
                "condition": "sunny",
                "temperature_celsius": 28,
                "humidity": 65
            },
            "milan": {
                "condition": "partly cloudy",
                "temperature_celsius": 24,
                "humidity": 70
            },
            "naples": {
                "condition": "sunny",
                "temperature_celsius": 30,
                "humidity": 60
            },
            "turin": {
                "condition": "rainy",
                "temperature_celsius": 18,
                "humidity": 85
            },
            "florence": {
                "condition": "hurricane",
                "temperature_celsius": 0,
                "humidity": 5
            }
        }
        
        # italian to english city name mapping
        city_translation = {
            "roma": "rome",
            "milano": "milan",
            "napoli": "naples",
            "torino": "turin",
            "firenze": "florence"
        }
        
        # default response for unknown locations
        weather_data = {
            "condition": "sunny",
            "temperature_celsius": 25,
            "humidity": 60
        }
        
        # normalize location for lookup
        location_lower = location.lower()
        
        # check if we need to translate the city name
        for italian_name, english_name in city_translation.items():
            if italian_name in location_lower:
                location_lower = english_name
                break
        
        # check if we have mock data for this location
        if location_lower in weather_conditions:
            weather_data = weather_conditions[location_lower]
        else:
            # try partial matching
            for city, data in weather_conditions.items():
                if city in location_lower:
                    weather_data = data
                    break
        
        # convert temperature if needed
        temp_celsius = weather_data["temperature_celsius"]
        temp = temp_celsius if unit == "celsius" else (temp_celsius * 9/5) + 32
        temp_unit = "°C" if unit == "celsius" else "°F"
        
        # italian weather condition translations
        condition_translation = {
            "sunny": "soleggiato",
            "partly cloudy": "parzialmente nuvoloso",
            "cloudy": "nuvoloso",
            "rainy": "piovoso",
            "snowy": "nevoso",
            "stormy": "tempestoso"
        }
        
        # translate condition to italian
        condition_italian = condition_translation.get(weather_data["condition"], weather_data["condition"])
        
        # format response
        response = (
            f"il meteo a {location} è attualmente {condition_italian}. "
            f"la temperatura è di {temp:.1f}{temp_unit} con umidità al {weather_data['humidity']}%."
        )
        
        # add detailed log for debugging
        logger.debug(f"Weather data found: {weather_data}")
        logger.debug(f"Formatted response: {response}")
        
        return response
        
    except Exception as e:
        logger.error(f"error getting weather: {str(e)}")
        return f"mi dispiace, ma non sono riuscito a ottenere le informazioni meteo per {location}."