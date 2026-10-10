#!/usr/bin/env python3
"""Erzeugt einmalig ein echtes Hillshade-Relief aus offenen Terrain-Höhendaten.
Keine Kachel-Anfragen durch Website-Besucher. Bei der Erzeugung online ausführen.
Quelle: Mapzen Terrain Tiles / Terrain data incl. Austria DGM, attribution beachten.
Benötigt: pip install requests pillow numpy
"""
import io, math
from pathlib import Path
import numpy as np
import requests
from PIL import Image

WEST, EAST, SOUTH, NORTH = 9.70, 13.10, 46.65, 47.85
ZOOM, TILE = 9, 256
WIDTH, HEIGHT = 1600, 1200
OUT = Path('tirol_relief.png')
BASE = 'https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png'

def world_px(lon,lat):
    lat=max(-85.05112878,min(85.05112878,lat))
    n=2**ZOOM*TILE
    return (lon+180)/360*n, (1-math.asinh(math.tan(math.radians(lat)))/math.pi)/2*n

left,top=world_px(WEST,NORTH)
right,bottom=world_px(EAST,SOUTH)
x0,y0=int(math.floor(left/TILE))-1,int(math.floor(top/TILE))-1
x1,y1=int(math.floor(right/TILE))+1,int(math.floor(bottom/TILE))+1
mosaic=np.empty(((y1-y0+1)*TILE,(x1-x0+1)*TILE),dtype=np.float32)
session=requests.Session()
for y in range(y0,y1+1):
    for x in range(x0,x1+1):
        url=BASE.format(z=ZOOM,x=x,y=y)
        response=session.get(url,timeout=35)
        response.raise_for_status()
        rgb=np.asarray(Image.open(io.BytesIO(response.content)).convert('RGB'),dtype=np.float32)
        elevations=rgb[:,:,0]*256+rgb[:,:,1]+rgb[:,:,2]/256-32768
        mosaic[(y-y0)*TILE:(y-y0+1)*TILE,(x-x0)*TILE:(x-x0+1)*TILE]=elevations
        print('Geladen:',ZOOM,x,y,flush=True)

# Pixelpositionen exakt in Web-Mercator interpolieren (wie Leaflet imageOverlay).
xx=np.linspace(left,right,WIDTH,endpoint=False,dtype=np.float64)+(right-left)/(2*WIDTH)
yy=np.linspace(top,bottom,HEIGHT,endpoint=False,dtype=np.float64)+(bottom-top)/(2*HEIGHT)
ix=np.clip(np.rint(xx-x0*TILE).astype(int),0,mosaic.shape[1]-1)
iy=np.clip(np.rint(yy-y0*TILE).astype(int),0,mosaic.shape[0]-1)
z=mosaic[np.ix_(iy,ix)]
# Meter/Pixel am Kartenmittelpunkt, approximiert in lokalen Ost-/Nordrichtungen.
m_per_lon=111320*math.cos(math.radians((SOUTH+NORTH)/2))
m_per_lat=111320
sx=((EAST-WEST)*m_per_lon)/WIDTH
sy=((NORTH-SOUTH)*m_per_lat)/HEIGHT
grad_y,grad_x=np.gradient(z,sy,sx)
# Lambert-Beleuchtung: Sonne aus Nordwesten, Höhe 45°.
azimuth=math.radians(315)
altitude=math.radians(45)
nx=-grad_x; ny=grad_y; nz=np.ones_like(z)
norm=np.sqrt(nx*nx+ny*ny+nz*nz)
light=(nx*math.sin(azimuth)*math.cos(altitude)+ny*math.cos(azimuth)*math.cos(altitude)+nz*math.sin(altitude))/norm
shade=np.clip(0.38+0.58*light,0,1)
# zurückhaltende helle Grau-Relief-Karte unter halbtransparenten Wetterfarben
v=np.clip(225-80*(1-shade)-12*np.clip(z,0,3500)/3500,105,236).astype(np.uint8)
Image.fromarray(np.stack([v,v,v],axis=-1),'RGB').save(OUT,optimize=True)
print(f'Fertig: {OUT} ({OUT.stat().st_size/1024/1024:.1f} MB)')
print('Datenquelle: Mapzen Terrain Tiles (u.a. © offene Daten Österreichs – DGM Österreich).')
