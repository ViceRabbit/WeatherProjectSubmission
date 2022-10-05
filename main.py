import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import asyncio
import aiohttp
from citipy import citipy
import time
import pandas as pd
import json

# API Example:
# https://api.openweathermap.org/data/2.5/forecast?q=city&cnt=1&units=imperial&appid=APIKEY


# weather_key = os.environ['weather_key'] - API Key stored in environment variables
weather_key = 'REPLACE WITH APIKEY FOR CODE TO WORK'

baselink = 'https://api.openweathermap.org/data/2.5/forecast?q='
apiEndLink = f'&cnt=1&units=imperial&appid={weather_key}'
sendAPI = lambda name: f'{baselink}{name}{apiEndLink}'

sample = 1200
coordinatesLGLT = zip(np.random.uniform(low = -90.000, high = 90.000, size=sample), np.random.uniform(low=-180.000, high=180.000, size=sample))
print('Generated sample of', sample)

confirmed_cities = {citipy.nearest_city(cordx[0], cordx[1]).city_name for cordx in tuple(coordinatesLGLT)}

InformationDict = dict()

InformationDict['Latitude'] = {}
InformationDict['Longitude'] = {}
InformationDict['Temperature'] = {}
InformationDict['Max Temp'] = {}
InformationDict['Humidity'] = {}
InformationDict['Cloudiness'] = {}
InformationDict['Wind Speed'] = {}

# Extracts Json and archives to InformationDict
def JsonExtractAPI(JsonArray):
  # Formatted = json.dumps(JsonArray, indent=2, sort_keys=True)
  try:
    CityJsonName = JsonArray["city"]["name"].split('/')[0]
    # .split because ansi escape sequence can appear
  except KeyError:
    print(f'JSON ERROR! Json: {json.dumps(JsonArray, indent=2, sort_keys=True)}')
  JsonLat = JsonArray["city"]["coord"]["lat"]
  JsonLon = JsonArray['city']['coord']['lon']
  JsonTemp = JsonArray ["list"][0]['main']["temp"]
  JsonMaxTemp = JsonArray['list'][0]['main']['temp_max']
  JsonHumidity = JsonArray["list"][0]['main']['humidity']
  JsonCloudiness = JsonArray["list"][0]['clouds']['all']
  JsonWindSpeed = JsonArray['list'][0]['wind']['speed']

  InformationDict['Latitude'][CityJsonName] = JsonLat
  InformationDict['Longitude'][CityJsonName] = JsonLon
  InformationDict['Temperature'][CityJsonName] = JsonTemp
  InformationDict['Max Temp'][CityJsonName] = JsonMaxTemp
  InformationDict['Humidity'][CityJsonName] = JsonHumidity
  InformationDict['Cloudiness'][CityJsonName] = JsonCloudiness
  InformationDict['Wind Speed'][CityJsonName] = JsonWindSpeed

start = time.time()

def req_tasks(session):
  tasks = []
  for city in confirmed_cities:
    tasks.append(session.get(sendAPI(city), ssl=False))
  return tasks

async def RequestAPI(): # Async Function; aiohttp for multi-tasked reqs
  async with aiohttp.ClientSession() as s: # session for optimization
    tasks = req_tasks(s)
    responses = await asyncio.gather(*tasks) # organizes reqs to exec
    for response in responses:
      # 'city not found' is a common error with the API: 
      if not 'city not found' in await response.text():
        JsonExtractAPI(await response.json())

asyncio.run(RequestAPI())

end = time.time()
print('Entire bulk of API Calls took', (end-start))

DataframeDict = pd.DataFrame.from_dict(InformationDict)
DataframeDict.to_csv('OutputCityData/cities.csv')
PlotLatVals = tuple(InformationDict['Latitude'].values())
PlotTempVals = tuple(InformationDict['Temperature'].values())
PlotHumVals = tuple(InformationDict['Humidity'].values())
PlotCloudVals = tuple(InformationDict['Cloudiness'].values())
PlotWindVals = tuple(InformationDict['Wind Speed'].values())

# creating/designating sub-plots for figure
fig, axs = plt.subplots(2, 2)
VsTempAx = axs[0, 0]
VsHumAx = axs[0, 1]
VsCloudAx = axs[1, 0]
VsWindAx = axs[1, 1]

VsTempAx.scatter(PlotLatVals, PlotTempVals, c='red')
VsTempAx.set_title('City Latitude vs City Temperature')
VsTempAx.set_xlabel('Latitude')
VsTempAx.set_ylabel('Temperature')

VsHumAx.scatter(PlotLatVals, PlotHumVals, c='darkkhaki')
VsHumAx.set_title('City Latitude vs City Humidity')
VsHumAx.set_xlabel('Latitude')
VsHumAx.set_ylabel('Humidity')

VsCloudAx.scatter(PlotLatVals, PlotCloudVals, c='gray')
VsCloudAx.set_title('City Latitude vs City Cloudiness')
VsCloudAx.set_xlabel('Latitude')
VsCloudAx.set_ylabel('Cloudiness')

VsWindAx.scatter(PlotLatVals, PlotWindVals, c='powderblue')
VsWindAx.set_title('City Latitude vs City Wind Speed')
VsWindAx.set_xlabel('Latitude')
VsWindAx.set_ylabel('Wind Speed')

fig.tight_layout()
fig.savefig('OutputCityData/StatisticsResults.pdf')
print('Saved figure as StatisticResults.pdf in OutputCityData!')

SouthernCities = [city for city in InformationDict['Latitude'] if InformationDict['Latitude'][city] < 0]

NorthernCities = [city for city in InformationDict['Latitude'] if InformationDict['Latitude'][city] >= 0]

# Linear Regression Function
def PlotLinearRegression(NorthAndSouthX, NorthAndSouthY, label):
  fig2, axs2 = plt.subplots(2, 1)
  SouthAx = axs2[1]
  NorthAx = axs2[0]

  nx, ny, sx, sy = NorthAndSouthX[0], NorthAndSouthY[0], NorthAndSouthX[1], NorthAndSouthY[1]
  numsx = np.array(sx)
  numsy = np.array(sy)
  numnx = np.array(nx)
  numny = np.array(ny)

  res_s = stats.linregress(sx, sy)
  res_n = stats.linregress(nx, ny)
  slope_s = res_s.slope
  slope_n = res_n.slope
  intercept_s = res_s.intercept
  intercept_n = res_n.intercept

  ResResultS = (numsx*slope_s) + intercept_s
  ResResultN = (numnx*slope_n) + intercept_n

  SouthAx.set_title('Southern Hemisphere')
  SouthAx.set_xlabel('Latitude')
  SouthAx.set_ylabel(label)
  SouthAx.scatter(numsx, numsy, c='red')
  SouthAx.plot(numsx, ResResultS, c='blue')
  
  NorthAx.set_title('Northern Hermisphere')
  NorthAx.set_xlabel('Latitude')
  NorthAx.set_ylabel(label)
  NorthAx.scatter(numnx, numny, c='blue')
  NorthAx.plot(numnx, ResResultN, c='red')

  fig2.tight_layout()
  fig2.savefig(f'OutputCityData/LatitudeVs{label}Hemisphere.pdf')
  print(f'Saved figure as LatitudeVs{label}Hemisphere.pdf in OutputCityData!')

ReturnHemis = lambda dictName: ([InformationDict[dictName][city] for city in NorthernCities], [InformationDict[dictName][city] for city in SouthernCities])

LatitudeReg = ReturnHemis('Latitude')

# Plot Max Temp and Latitude (Both North and South Hemisphere)
MaxTempReg = ReturnHemis('Max Temp')
PlotLinearRegression(LatitudeReg, MaxTempReg, 'Max Temperature')

# Plot Humidity and Latitude (Both North And South hemisphere)
HumidityReg = ReturnHemis('Humidity')
PlotLinearRegression(LatitudeReg, HumidityReg, 'Humidity')

# Plot Cloudiness and Latitude (Both North And South hemisphere)
CloudReg = ReturnHemis('Cloudiness')
PlotLinearRegression(LatitudeReg, CloudReg, 'Cloudiness')

# Plot Wind Speed and Latitude (Both North And South hemisphere)
WindReg = ReturnHemis('Wind Speed')
PlotLinearRegression(LatitudeReg, WindReg, 'Wind Speed')

print('TIP: Move around the windows of figures to reveal others.')
plt.show()
