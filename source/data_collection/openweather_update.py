from core import get_configuration
from openweather_to_json import WeatherSource

application = 'weather_forecast'
# UNPSJB geo coordinates
latitude = '-45.824343'  
longitude = '-67.461986' 

raw_data = Path('~', 'Data', application, 'raw_data')

configuration = get_configuration(['API_KEY'])
source = WeatherSource(configuration['API_KEY'])
event = source.get(latitude, longitude)
source.print(event)
