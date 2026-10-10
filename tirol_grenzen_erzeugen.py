#!/usr/bin/env python3
"""Einmalig Tirols Landes-/Bezirksgrenzen aus GeoSphere Austria laden.
Quelle: GeoSphere Austria / Statistik Austria, Verwaltungsgrenzen Stand Februar 2025.
Ausgabe: zwei vereinfachte GeoJSON-Dateien, zum lokalen Hosting.
"""
import json
from pathlib import Path
import requests
from shapely.geometry import shape, mapping, Point
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
# Tirol besteht aus zwei räumlich getrennten Landesteilen: Nordtirol und Osttirol.
# Die Quelle liefert diese unter Umständen als ZWEI getrennte Features.
# Daher beide unabhängig über sichere Referenzpunkte bestimmen.
reference_points = {
    'Nordtirol (Innsbruck)': Point(11.404, 47.269),
    'Osttirol (Lienz)': Point(12.769, 46.828),
}
parts = []
for label, point in reference_points.items():
    matches = []
    for feature in states:
        geometry = shape(feature['geometry'])
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
        if geometry.covers(point):
            matches.append(geometry)
    if len(matches) != 1:
        raise RuntimeError(
            f'{label}: erwartet genau eine Landesfläche, gefunden {len(matches)}. '
            'Quell-Daten prüfen.'
        )
    parts.append(matches[0])
    print(f'{label}: Fläche gefunden', flush=True)

# Bei separaten Features entsteht MultiPolygon; bei gemeinsamer Geometrie
# wird diese durch unary_union nicht doppelt ausgegeben.
tirol = unary_union(parts)
if not tirol.is_valid:
    tirol = tirol.buffer(0)
if tirol.is_empty:
    raise RuntimeError('Leere Tirol-Geometrie')
for label, point in reference_points.items():
    if not tirol.covers(point):
        raise RuntimeError(f'{label} fehlt nach dem Zusammenführen')

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
if len(selected) != 9:
    raise RuntimeError(f'Erwarte 9 Tiroler Bezirke einschließlich Lienz, gefunden {len(selected)}; bitte Quelle prüfen')
# Absicherung: Der Bezirk Lienz muss enthalten sein.
if not any(g.covers(reference_points['Osttirol (Lienz)']) for g in selected):
    raise RuntimeError('Bezirk Lienz fehlt in den Bezirksgrenzen')

def write(filename,geom_list):
    obj={'type':'FeatureCollection','features':[{'type':'Feature','properties':{},'geometry':mapping(g)} for g in geom_list]}
    p=OUT/filename
    p.write_text(json.dumps(obj,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    print(f'{p}: {p.stat().st_size/1024:.1f} KB; {len(geom_list)} Objekte',flush=True)

write('tirol_landesgrenze.geojson',[tirol_simple])
write('tirol_bezirksgrenzen.geojson',selected)
print('Quellenangabe: © Statistik Austria / GeoSphere Austria, Verwaltungsgrenzen 2025')
