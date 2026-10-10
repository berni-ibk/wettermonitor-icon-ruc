#!/usr/bin/env python3
"""
Erzeugt einmalig ein deutlich verbessertes Tirol-Reliefbild
für die ICON-D2-RUC-Prognosekarten.

Ziel:
- eigenes Reliefbild ohne externe Kartenkachelaufrufe durch Website-Besucher
- deutlich plastischer als die bisherige sehr flache Version
- höhere Auflösung
- bilineare Resampling-Qualität statt grober Punktabtastung
- multidirektionales Hillshade für natürlichere Geländestruktur

Benötigt:
    pip install requests pillow numpy
"""

import io
import math
from pathlib import Path

import numpy as np
import requests
from PIL import Image

# ------------------------------------------------------------
# BEREICH TIROL
# ------------------------------------------------------------
WEST, EAST, SOUTH, NORTH = 9.70, 13.10, 46.65, 47.85

# ------------------------------------------------------------
# AUSGABE / DATENQUELLE
# ------------------------------------------------------------
ZOOM = 10
TILE = 256
WIDTH = 2200
HEIGHT = 1500
OUT = Path("tirol_relief.png")

# Terrarium-Höhendaten
BASE = "https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{z}/{x}/{y}.png"


# ------------------------------------------------------------
# HILFSFUNKTIONEN
# ------------------------------------------------------------
def world_px(lon, lat):
    """Lon/Lat -> WebMercator-Pixelkoordinaten."""
    lat = max(-85.05112878, min(85.05112878, lat))
    n = 2 ** ZOOM * TILE
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def load_tile(session, x, y):
    """Lädt eine Terrarium-Kachel und dekodiert Höhenmeter."""
    url = BASE.format(z=ZOOM, x=x, y=y)
    response = session.get(url, timeout=40)
    response.raise_for_status()

    rgb = np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"), dtype=np.float32)

    # Terrarium-Format:
    # height = (R * 256 + G + B / 256) - 32768
    elev = rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0 - 32768.0
    return elev.astype(np.float32)


def bilinear_sample(grid, x_coords, y_coords):
    """
    Bilineares Resampling aus dem DEM-Mosaik.
    x_coords und y_coords sind 1D-Koordinaten im Pixelraum des Mosaiks.
    """
    x0 = np.floor(x_coords).astype(int)
    y0 = np.floor(y_coords).astype(int)

    x1 = np.clip(x0 + 1, 0, grid.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, grid.shape[0] - 1)

    x0 = np.clip(x0, 0, grid.shape[1] - 1)
    y0 = np.clip(y0, 0, grid.shape[0] - 1)

    dx = (x_coords - x0)[None, :]
    dy = (y_coords - y0)[:, None]

    q11 = grid[np.ix_(y0, x0)]
    q21 = grid[np.ix_(y0, x1)]
    q12 = grid[np.ix_(y1, x0)]
    q22 = grid[np.ix_(y1, x1)]

    return (
        q11 * (1.0 - dx) * (1.0 - dy)
        + q21 * dx * (1.0 - dy)
        + q12 * (1.0 - dx) * dy
        + q22 * dx * dy
    )


def hillshade(grad_x, grad_y, azimuth_deg, altitude_deg, vert_exag=1.55):
    """
    Lambert-Schummerung aus Geländegradienten.
    Mehrere Richtungen werden später gemischt.
    """
    gx = grad_x * vert_exag
    gy = grad_y * vert_exag

    az = math.radians(azimuth_deg)
    alt = math.radians(altitude_deg)

    nx = -gx
    ny = gy
    nz = np.ones_like(gx)

    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    light = (
        nx * math.sin(az) * math.cos(alt)
        + ny * math.cos(az) * math.cos(alt)
        + nz * math.sin(alt)
    ) / norm

    return np.clip(light, 0.0, 1.0)


# ------------------------------------------------------------
# HAUPTTEIL
# ------------------------------------------------------------
def main():
    print("Berechne WebMercator-Ausschnitt ...", flush=True)

    left, top = world_px(WEST, NORTH)
    right, bottom = world_px(EAST, SOUTH)

    x0 = int(math.floor(left / TILE)) - 1
    y0 = int(math.floor(top / TILE)) - 1
    x1 = int(math.floor(right / TILE)) + 1
    y1 = int(math.floor(bottom / TILE)) + 1

    tile_cols = x1 - x0 + 1
    tile_rows = y1 - y0 + 1

    print(f"Kachelbereich z={ZOOM}: x={x0}..{x1}, y={y0}..{y1}", flush=True)

    mosaic = np.empty((tile_rows * TILE, tile_cols * TILE), dtype=np.float32)

    session = requests.Session()

    for ty, y in enumerate(range(y0, y1 + 1)):
        for tx, x in enumerate(range(x0, x1 + 1)):
            elev = load_tile(session, x, y)
            mosaic[
                ty * TILE : (ty + 1) * TILE,
                tx * TILE : (tx + 1) * TILE
            ] = elev
            print(f"Geladen: z={ZOOM} x={x} y={y}", flush=True)

    print("Resampling auf Zielauflösung ...", flush=True)

    xx = np.linspace(left, right, WIDTH, endpoint=False, dtype=np.float64)
    yy = np.linspace(top, bottom, HEIGHT, endpoint=False, dtype=np.float64)

    xx += (right - left) / (2.0 * WIDTH)
    yy += (bottom - top) / (2.0 * HEIGHT)

    # Lokale Pixelkoordinaten innerhalb des Mosaiks
    x_local = xx - x0 * TILE
    y_local = yy - y0 * TILE

    dem = bilinear_sample(mosaic, x_local, y_local)

    print("Hillshade berechnen ...", flush=True)

    mean_lat = (SOUTH + NORTH) / 2.0
    m_per_lon = 111320.0 * math.cos(math.radians(mean_lat))
    m_per_lat = 111320.0

    sx = ((EAST - WEST) * m_per_lon) / WIDTH
    sy = ((NORTH - SOUTH) * m_per_lat) / HEIGHT

    grad_y, grad_x = np.gradient(dem, sy, sx)

    # Mehrere Lichtquellen mischen -> natürlicheres Relief
    shade_nw = hillshade(grad_x, grad_y, 315, 48, 1.60)
    shade_ne = hillshade(grad_x, grad_y, 35, 52, 1.35)
    shade_sw = hillshade(grad_x, grad_y, 235, 32, 1.10)

    shade = 0.58 * shade_nw + 0.27 * shade_ne + 0.15 * shade_sw

    # Kontrast normalisieren
    p2, p98 = np.percentile(shade, [2, 98])
    if p98 > p2:
        shade = np.clip((shade - p2) / (p98 - p2), 0.0, 1.0)

    # Gamma etwas Richtung plastischer
    shade = shade ** 0.92

    print("Graubild erzeugen ...", flush=True)

    # leichte Höhenmodulation
    elev_norm = np.clip((dem - 350.0) / 3200.0, 0.0, 1.0)

    # helles Grundrelief, aber deutlich sichtbarer als bisher
    base = 243.0 - 16.0 * elev_norm

    # plastischere Schattierung
    relief = base - 82.0 * (1.0 - shade) + 7.0 * shade

    # Begrenzen
    relief = np.clip(relief, 138.0, 247.0).astype(np.uint8)

    rgb = np.stack([relief, relief, relief], axis=-1)

    img = Image.fromarray(rgb, "RGB")
    img.save(OUT, optimize=True)

    print(f"Fertig: {OUT} ({OUT.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
    print("Datenquelle: Mapzen Terrain Tiles / Terrarium, offene Höhendaten.", flush=True)


if __name__ == "__main__":
    main()
