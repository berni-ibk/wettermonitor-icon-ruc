#!/usr/bin/env python3
"""Einmalig Tirols Landes-/Bezirksgrenzen aus GeoSphere Austria laden.
Quelle: GeoSphere Austria / Statistik Austria, Verwaltungsgrenzen Stand Februar 2025.
Ausgabe: zwei vereinfachte GeoJSON-Dateien, zum lokalen Hosting.
"""
import json
from pathlib import Path
import requests
from shapely.geometry import shape, mapping
from shapely.ops import unary_union

BASE = 'https://gis.geosphere.at/maps/rest/services/grenzen/admin_grenzen_oesterreich/MapServer'
OUT = Path('grenzen_tirol')
OUT.mkdir(exist_ok=True)

def load(layer):
    url=f'{BASE}/{layer}/query'
    params={'where':'1=1','outFields':'*','returnGeometry':'true','outSR':'4326','f':'geojson'}
    r=requests.get(url,params=params,timeout=90,headers={'User-Agent':'Wettermonitor-Tirol-Grenzen/1.0'})
    r.raise_for_status()
    data=r.json()
    if data.get('type') != 'FeatureCollection' or not data.get('features'):
        raise RuntimeError(f'Layer {layer}: leere/fehlerhafte Antwort {str(data)[:300]}')
    print(f'Layer {layer}: {len(data["features"])} Features',flush=True)
    return data['features']

states=load(0)
matching=[]
for feature in states:
    props=feature.get('properties') or {}
    if any('tirol' in str(v).casefold() for v in props.values()):
        matching.append(feature)
if len(matching)!=1:
    raise RuntimeError(f'Erwarte genau ein Bundesland Tirol; gefunden {len(matching)}. Attribute: {[x.get("properties") for x in states[:2]]}')
tirol=shape(matching[0]['geometry'])
if tirol.is_empty or not tirol.is_valid:
    tirol=tirol.buffer(0)
# Gut für Übersichtskarten, auch auf Smartphones: weniger Daten ohne sichtbare Unterschiede.
tirol_simple=tirol.simplify(0.0008,preserve_topology=True)
districts=load(1)
selected=[]
for feat in districts:
    geom=shape(feat['geometry'])
    inter=geom.intersection(tirol)
    if inter.is_empty:
        continue
    if inter.area < geom.area * 0.5:
        continue
    selected.append(inter.simplify(0.0008,preserve_topology=True))
if len(selected)<8 or len(selected)>10:
    raise RuntimeError(f'Erwarte 9 Tiroler Bezirke, gefunden {len(selected)}; bitte Quelle prüfen')

def write(filename,geom_list):
    obj={'type':'FeatureCollection','features':[{'type':'Feature','properties':{},'geometry':mapping(g)} for g in geom_list]}
    p=OUT/filename
    p.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print(f'{p}: {p.stat().st_size/1024:.1f} KB; {len(geom_list)} Objekte',flush=True)

write('tirol_landesgrenze.geojson',[tirol_simple])
write('tirol_bezirksgrenzen.geojson',selected)
print('Quellenangabe: © Statistik Austria / GeoSphere Austria, Verwaltungsgrenzen 2025')
