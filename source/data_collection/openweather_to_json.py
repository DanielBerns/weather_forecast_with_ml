import requests                 
import time                     


indentation = ' '


class WeatherSource:
    def __init__(self, api_key):
        self._api_key = api_key
        
    @property
    def api_key(self):
        return self._api_key
    
    def get(self, latitude, longitude):
        url = ''.join(
            [f"https://api.openweathermap.org/data/2.5/weather?",
             f"lat={latitude:s}",
             f"&lon={longitude:s}",
             f"&appid={self._api_key:s}",
             "&units=metric"])

        response = requests.get(url)
        event = {}
        if response.status_code == 200:
            as_json = response.json()                # json 
            last_update = time.gmtime(as_json["dt"]) # check: UNIX to UTC
            event["sample"] = as_json
            event["message"] = f"http code: 200"
            event["ok"] = True
        else:
            event["message"] = f"http code: {response.status_code:d}"
            event["ok"] = False
        return event

    def print(self, event):
        # https://openweathermap.org/weather-data
        sample = event['sample']
        coord = sample['coord']
        print('lat', coord['lat'], 'lon', coord['lon'])
        weather = sample['weather'][0]
        print('id', weather['id'])
        print('main', weather['main'])
        print('description', weather['description'])
        print('icon', weather['icon'])
        main = sample['main']
        print('temp', main['temp'])
        print('feels_like', main['feels_like'])
        print('temp_min', main['temp_min'])
        print('temp_max', main['temp_max'])
        print('pressure', main['pressure'])
        print('humidity', main['humidity'])
        visibility = sample['visibility']
        print('visibility', visibility)
        wind = sample['wind']
        print('wind_speed', wind['speed'])
        print('wind_deg', wind['deg'])
        clouds = sample['clouds']
        for k, v in clouds.items():
            print(k, v)
        dt = sample['dt']
        print(dt)
        _sys = sample['sys']
        for k, v in _sys.items():
            print('sys', k, v)
        timezone = sample['timezone']
        print('timezone', timezone)
        _id = sample['id']
        print('timezone_id', _id)
        name = sample['name']
        print('timezone_name', name)
        cod = sample['cod']
        print('timezone_cod', cod)
        
