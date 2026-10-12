# ============================================================
# WETTERMONITOR.AT – ICON-D2-RUC TIROL
# 27 Prognosestunden, 9 Parameter einschließlich Niederschlagssumme, Bewölkung, Böen und Schnee
# + echte interpolierte Modellwerte für Klick-Popups
# Ausführung auf GitHub Actions (keine Colab-Abhängigkeit)
# ============================================================
import sys, subprocess, importlib.util, re, json, bz2, gc, shutil, hashlib, html
from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.parse import quote, unquote

PACKAGES = {'numpy':'numpy','scipy':'scipy','matplotlib':'matplotlib','requests':'requests','PIL':'pillow','eccodes':'eccodes','netCDF4':'netCDF4'}
missing = [package for module,package in PACKAGES.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable,'-m','pip','install','-q',*missing])

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.colorbar import ColorbarBase
from PIL import Image
from netCDF4 import Dataset
from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release

# Kartengrenzen identisch zum funktionierenden Viewer
LON_MIN,LON_MAX = 9.70,13.10
LAT_MIN,LAT_MAX = 46.65,47.85
GRID_MARGIN = 0.15
WIDTH,HEIGHT = 1600,1200
VALUE_WIDTH,VALUE_HEIGHT = 800,600
HOURS = list(range(1,28))

# Standorte, deren interpolierte RUC-Modellwerte zusätzlich direkt
# in die Metadaten geschrieben werden. Diese Punktwerte werden vom
# Modelllauf-Archiv verwendet; die Karten-/PNG-Ausgabe bleibt unverändert.
POINT_LOCATIONS = {
    'sieglanger': {
        'name': 'Sieglanger / Innsbruck-West',
        'lat': 47.2602,
        'lon': 11.344,
    },
    'innsbruck_flughafen': {
        'name': 'Innsbruck Flughafen',
        'lat': 47.2600,
        'lon': 11.35667,
    },
}
DWD_ROOT = 'https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc/p/'
GRID_URL = 'https://opendata.dwd.de/weather/lib/cdo/icon_grid_0047_R19B07_L.nc.bz2'
OUT = Path('output')
TMP = Path('temp')
CACHE = Path('cache')
for folder in (OUT,TMP,CACHE): folder.mkdir(parents=True,exist_ok=True)
for p in OUT.iterdir():
    if p.is_file(): p.unlink()
GRIB = TMP/'daten.grib2'
GRID_BZ2 = TMP/'grid.nc.bz2'
GRID_NC = TMP/'grid.nc'
PARAMS = {
 'temperature_2m':{'dwd':'T_2M','name':'Temperatur 2 m','unit':'°C','short_names':('2t','T_2M'),'limits':(-5,25),'cmap':'turbo','legend':'icon_d2_ruc_legende.png','decimals':1},
 'precipitation':{'dwd':'TOT_PREC','name':'Niederschlag 1 h','unit':'mm','short_names':('tp','TOT_PREC'),'limits':(0,20),'cmap':'ruc_blau','legend':'icon_d2_ruc_niederschlag_legende.png','decimals':2},
 'precipitation_sum':{'dwd':'TOT_PREC','name':'Niederschlagssumme','unit':'mm','short_names':('tp','TOT_PREC'),'limits':(0,50),'cmap':'ruc_blau','legend':'icon_d2_ruc_niederschlagssumme_legende.png','decimals':2},
 'cloud_cover':{'dwd':'CLCT','name':'Bewölkung','unit':'%','short_names':('tcc','CLCT'),'limits':(0,100),'cmap':'cloud_gray','legend':'icon_d2_ruc_bewoelkung_legende.png','decimals':0},
 'cloud_low':{'dwd':'CLCL','name':'Bewölkung tief','unit':'%','short_names':('lcc','CLCL'),'limits':(0,100),'cmap':'cloud_gray','legend':'icon_d2_ruc_cloud_low_legende.png','decimals':0},
 'cloud_mid':{'dwd':'CLCM','name':'Bewölkung mittel','unit':'%','short_names':('mcc','CLCM'),'limits':(0,100),'cmap':'cloud_gray','legend':'icon_d2_ruc_cloud_mid_legende.png','decimals':0},
 'cloud_high':{'dwd':'CLCH','name':'Bewölkung hoch','unit':'%','short_names':('hcc','CLCH'),'limits':(0,100),'cmap':'cloud_gray','legend':'icon_d2_ruc_cloud_high_legende.png','decimals':0},
 'wind_gusts':{'dwd':'VMAX_10M','name':'Windböen','unit':'km/h','short_names':('10fg','VMAX_10M','gust'),'limits':(0,120),'cmap':'gust_ruc','legend':'icon_d2_ruc_boen_legende.png','decimals':1},
 'snowfall':{'dwd':'SNOW_GSP','name':'Schneefall (Wasseräquivalent) 1 h','unit':'mm','short_names':('sf','SNOW_GSP','asnow'),'limits':(0,10),'cmap':'snow_ruc','legend':'icon_d2_ruc_schneefall_legende.png','decimals':2}
}


# Feste Farbstufen: dieselben Grenzen und Farben wie im HTML-Viewer.
# Farbübergänge erfolgen sprunghaft, Modell- und Klickwerte bleiben unverändert.
COLOR_LEVELS = {
    'temperature_2m': [-5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0],
    'precipitation': [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20],
    'precipitation_sum': [0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50],
    'cloud_cover': [0.0, 8.33333, 16.66667, 25.0, 33.33333, 41.66667, 50.0, 58.33333, 66.66667, 75.0, 83.33333, 91.66667, 100.0],
    'cloud_low': [0.0, 8.33333, 16.66667, 25.0, 33.33333, 41.66667, 50.0, 58.33333, 66.66667, 75.0, 83.33333, 91.66667, 100.0],
    'cloud_mid': [0.0, 8.33333, 16.66667, 25.0, 33.33333, 41.66667, 50.0, 58.33333, 66.66667, 75.0, 83.33333, 91.66667, 100.0],
    'cloud_high': [0.0, 8.33333, 16.66667, 25.0, 33.33333, 41.66667, 50.0, 58.33333, 66.66667, 75.0, 83.33333, 91.66667, 100.0],
    'wind_gusts': [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
    'snowfall': [0, 0.05, 0.1, 0.2, 0.4, 0.6, 1, 1.5, 2.5, 4, 6, 8, 10],
}
COLOR_STEPS = {
    'temperature_2m': ['#3b2f80', '#466be3', '#3ba0fd', '#1ad2d2', '#32f298', '#80ff53', '#bef434', '#eecf3a', '#fe9e2f', '#f26014', '#d02f05', '#9b0f01'],
    'precipitation': ['#8ac9f3', '#65b8eb', '#43a4e0', '#268fd3', '#167dc8', '#086bbb', '#075ca8', '#075099', '#06458b', '#053a7a', '#032e65', '#021c47'],
    'precipitation_sum': ['#8ac9f3', '#65b8eb', '#43a4e0', '#268fd3', '#167dc8', '#086bbb', '#075ca8', '#075099', '#06458b', '#053a7a', '#032e65', '#021c47'],
    'cloud_cover': [(0.7, 0.73, 0.77, 0.045), (0.66, 0.69, 0.73, 0.09), (0.61, 0.64, 0.68, 0.15), (0.56, 0.59, 0.63, 0.22), (0.51, 0.54, 0.58, 0.3), (0.46, 0.49, 0.53, 0.39), (0.42, 0.45, 0.49, 0.48), (0.38, 0.41, 0.45, 0.57), (0.34, 0.37, 0.41, 0.66), (0.3, 0.33, 0.37, 0.74), (0.26, 0.29, 0.33, 0.81), (0.23, 0.26, 0.3, 0.87)],
    'cloud_low': [(0.7, 0.73, 0.77, 0.045), (0.66, 0.69, 0.73, 0.09), (0.61, 0.64, 0.68, 0.15), (0.56, 0.59, 0.63, 0.22), (0.51, 0.54, 0.58, 0.3), (0.46, 0.49, 0.53, 0.39), (0.42, 0.45, 0.49, 0.48), (0.38, 0.41, 0.45, 0.57), (0.34, 0.37, 0.41, 0.66), (0.3, 0.33, 0.37, 0.74), (0.26, 0.29, 0.33, 0.81), (0.23, 0.26, 0.3, 0.87)],
    'cloud_mid': [(0.7, 0.73, 0.77, 0.045), (0.66, 0.69, 0.73, 0.09), (0.61, 0.64, 0.68, 0.15), (0.56, 0.59, 0.63, 0.22), (0.51, 0.54, 0.58, 0.3), (0.46, 0.49, 0.53, 0.39), (0.42, 0.45, 0.49, 0.48), (0.38, 0.41, 0.45, 0.57), (0.34, 0.37, 0.41, 0.66), (0.3, 0.33, 0.37, 0.74), (0.26, 0.29, 0.33, 0.81), (0.23, 0.26, 0.3, 0.87)],
    'cloud_high': [(0.7, 0.73, 0.77, 0.045), (0.66, 0.69, 0.73, 0.09), (0.61, 0.64, 0.68, 0.15), (0.56, 0.59, 0.63, 0.22), (0.51, 0.54, 0.58, 0.3), (0.46, 0.49, 0.53, 0.39), (0.42, 0.45, 0.49, 0.48), (0.38, 0.41, 0.45, 0.57), (0.34, 0.37, 0.41, 0.66), (0.3, 0.33, 0.37, 0.74), (0.26, 0.29, 0.33, 0.81), (0.23, 0.26, 0.3, 0.87)],
    'wind_gusts': ['#e9f5dc', '#d3e99c', '#a8d453', '#f2e45e', '#ffc344', '#ffa132', '#f7832e', '#e86428', '#d4413b', '#b92b59', '#913074', '#662381'],
