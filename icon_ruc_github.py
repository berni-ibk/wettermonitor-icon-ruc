
# ============================================================
# WETTERMONITOR.AT – ICON-D2-RUC TIROL
# 12 Prognosestunden, Temperatur, Niederschlag 1 h,
# Niederschlagssumme und Bewölkung
# + echte interpolierte Modellwerte beim Kartenklick
# Ausführung über GitHub Actions
# ============================================================

import sys
import subprocess
import importlib.util
import re
import json
import bz2
import gc
import shutil
import hashlib
import html

from pathlib import Path
from datetime import datetime, timezone, timedelta
from urllib.parse import quote, unquote

PACKAGES = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'requests': 'requests',
    'PIL': 'pillow',
    'eccodes': 'eccodes',
    'netCDF4': 'netCDF4'
}

missing = [
    package
    for module, package in PACKAGES.items()
    if importlib.util.find_spec(module) is None
]

if missing:
    subprocess.check_call([
        sys.executable,
        '-m',
        'pip',
        'install',
        '-q',
        *missing
    ])

import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from matplotlib import colormaps
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colorbar import ColorbarBase
from PIL import Image
from netCDF4 import Dataset
from eccodes import (
    codes_grib_new_from_file,
    codes_get,
    codes_get_array,
    codes_release
)

# ------------------------------------------------------------
# GRUNDEINSTELLUNGEN
# ------------------------------------------------------------

LON_MIN, LON_MAX = 9.70, 13.10
LAT_MIN, LAT_MAX = 46.65, 47.85

GRID_MARGIN = 0.15

WIDTH, HEIGHT = 1600, 1200
VALUE_WIDTH, VALUE_HEIGHT = 800, 600

HOURS = list(range(1, 13))

DWD_ROOT = (
    'https://opendata.dwd.de/weather/nwp/v1/m/'
    'icon-d2-ruc/p/'
)

GRID_URL = (
    'https://opendata.dwd.de/weather/lib/cdo/'
    'icon_grid_0047_R19B07_L.nc.bz2'
)

OUT = Path('output')
TMP = Path('temp')
CACHE = Path('cache')

for folder in (OUT, TMP, CACHE):
    folder.mkdir(parents=True, exist_ok=True)

for p in OUT.iterdir():
    if p.is_file():
        p.unlink()

GRIB = TMP / 'daten.grib2'
GRID_BZ2 = TMP / 'grid.nc.bz2'
GRID_NC = TMP / 'grid.nc'

# ------------------------------------------------------------
# WETTERPARAMETER
# ------------------------------------------------------------

PARAMS = {
    'temperature_2m': {
        'dwd': 'T_2M',
        'name': 'Temperatur 2 m',
        'unit': '°C',
        'short_names': ('2t', 'T_2M'),
        'limits': (-5, 25),
        'cmap': 'turbo',
        'legend': 'icon_d2_ruc_legende.png',
        'decimals': 1
    },

    'precipitation': {
        'dwd': 'TOT_PREC',
        'name': 'Niederschlag 1 h',
        'unit': 'mm',
        'short_names': ('tp', 'TOT_PREC'),
        'limits': (0, 20),
        'cmap': 'ruc_blau',
        'legend': 'icon_d2_ruc_niederschlag_legende.png',
        'decimals': 2
    },

    'precipitation_sum': {
        'dwd': 'TOT_PREC',
        'name': 'Niederschlagssumme',
        'unit': 'mm',
        'short_names': ('tp', 'TOT_PREC'),
        'limits': (0, 50),
        'cmap': 'ruc_blau',
        'legend': 'icon_d2_ruc_niederschlagssumme_legende.png',
        'decimals': 2
    },

    'cloud_cover': {
        'dwd': 'CLCT',
        'name': 'Bewölkung',
        'unit': '%',
        'short_names': ('tcc', 'CLCT'),
        'limits': (0, 100),
        'cmap': 'Greys',
        'legend': 'icon_d2_ruc_bewoelkung_legende.png',
        'decimals': 0
    }
}

# ------------------------------------------------------------
# NEUE NIEDERSCHLAGSFARBSKALA
# ------------------------------------------------------------
# Dezenter Verlauf von sehr hellem Blau zu tiefem Dunkelblau.
#
# 0 mm bleibt transparent.
# Bereits kleinste positive Mengen werden hellblau dargestellt.
# Steigende Mengen werden zunehmend dunkler.
#
# Die Modellwerte bleiben unverändert.
# ------------------------------------------------------------

RAIN_CMAP = LinearSegmentedColormap.from_list(
    'ruc_blau',
    [
        (0.00, '#dcefff'),
        (0.05, '#cfe8ff'),
        (0.10, '#bddfff'),
        (0.20, '#9fd0f7'),
        (0.35, '#7ebbe9'),
        (0.50, '#5ca3da'),
        (0.65, '#3f8bca'),
        (0.80, '#256fb2'),
        (0.92, '#14538f'),
        (1.00, '#0a3568')
    ],
    N=256
)


def get_cmap(param):
    if param in ('precipitation', 'precipitation_sum'):
        return RAIN_CMAP

    return colormaps[PARAMS[param]['cmap']]


# ------------------------------------------------------------
# DOWNLOAD
# ------------------------------------------------------------

def download(url, path, limit_mb):
    total = 0

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(30, 120)
        ) as r:

            r.raise_for_status()

            content_length = int(
                r.headers.get('Content-Length', '0')
            )

            if content_length > limit_mb * 1024**2:
                raise RuntimeError(
                    'Download überschreitet Größenlimit'
                )

            with open(path, 'wb') as f:
                for block in r.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not block:
                        continue

                    total += len(block)

                    if total > limit_mb * 1024**2:
                        raise RuntimeError(
                            'Download zu groß'
                        )

                    f.write(block)

        if total == 0:
            raise RuntimeError('Leerer Download')

        return total

    except Exception:
        path.unlink(missing_ok=True)
        raise


# ------------------------------------------------------------
# ICON-GITTER
# ------------------------------------------------------------

cache_id = hashlib.sha256(
    (
        f'{LON_MIN}_{LON_MAX}_'
        f'{LAT_MIN}_{LAT_MAX}_'
        f'{GRID_MARGIN}_{GRID_URL}'
    ).encode()
).hexdigest()[:16]

GRID_CACHE = CACHE / f'tirol_grid_{cache_id}.npz'


def create_grid_cache():
    print('ICON-Gitter herunterladen ...')

    try:
        download(GRID_URL, GRID_BZ2, 250)

        unpacked = 0

        with bz2.open(GRID_BZ2, 'rb') as src:
            with open(GRID_NC, 'wb') as dst:

                while True:
                    b = src.read(1024 * 1024)

                    if not b:
                        break

                    unpacked += len(b)

                    if unpacked > 600 * 1024**2:
                        raise RuntimeError(
                            'Entpacktes Gitter zu groß'
                        )

                    dst.write(b)

        with Dataset(GRID_NC) as ds:

            uuid = str(
                ds.getncattr('uuidOfHGrid')
            ).replace('-', '').lower()

            all_lon = np.degrees(
                np.asarray(
                    ds.variables['clon'][:]
                ).ravel()
            )

            all_lat = np.degrees(
                np.asarray(
                    ds.variables['clat'][:]
                ).ravel()
            )

        mask = (
            (all_lon >= LON_MIN - GRID_MARGIN) &
            (all_lon <= LON_MAX + GRID_MARGIN) &
            (all_lat >= LAT_MIN - GRID_MARGIN) &
            (all_lat <= LAT_MAX + GRID_MARGIN)
        )

        idx = np.where(mask)[0]

        if len(idx) < 100:
            raise RuntimeError(
                'Gitter unvollständig'
            )

        np.savez_compressed(
            GRID_CACHE,
            indices=idx.astype('int32'),
            lon=all_lon[idx],
            lat=all_lat[idx],
            grid_uuid=np.array(uuid),
            total_points=np.array(len(all_lon))
        )

    finally:
        GRID_BZ2.unlink(missing_ok=True)
        GRID_NC.unlink(missing_ok=True)


if not GRID_CACHE.exists():
    create_grid_cache()
else:
    print('Vorhandener Tirol-Gittercache wird verwendet.')

with np.load(GRID_CACHE) as d:

    grid_indices = d['indices'].astype('int64')
    lon = d['lon'].astype('float64')
    lat = d['lat'].astype('float64')

    grid_uuid = str(
        d['grid_uuid'].item()
    )

    total_points = int(
        d['total_points'].item()
    )

print(
    'ICON-Gitterpunkte für Tirol:',
    len(lon)
)


# ------------------------------------------------------------
# NEUESTEN MODELLLAUF ERMITTELN
# ------------------------------------------------------------

r = requests.get(
    DWD_ROOT + 'T_2M/r/',
    timeout=30
)

r.raise_for_status()

links = re.findall(
    r'href\s*=\s*["\']([^"\']+)["\']',
    r.text,
    flags=re.I
)

runs = sorted(
    {
        name
        for link in links
        if re.fullmatch(
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}',
            name := unquote(
                html.unescape(link)
            ).rstrip('/').split('/')[-1]
        )
    },
    reverse=True
)

if not runs:
    raise RuntimeError(
        'Keine DWD-Modellläufe gefunden'
    )

print(
    'Neuester gelisteter Modelllauf:',
    runs[0]
)


# ------------------------------------------------------------
# GRIB2 AUSLESEN
# ------------------------------------------------------------

def read_grib(path, param):

    with open(path, 'rb') as stream:

        gid = codes_grib_new_from_file(stream)

        if gid is None:
            raise RuntimeError(
                'GRIB2 nicht dekodierbar'
            )

        try:
            uuid = str(
                codes_get(gid, 'uuidOfHGrid')
            ).replace('-', '').lower()

            short = str(
                codes_get(gid, 'shortName')
            )

            units = str(
                codes_get(gid, 'units')
            )

            point_count = int(
                codes_get(gid, 'numberOfDataPoints')
            )

            raw = codes_get_array(
                gid, 'values'
            )

            missing_value = float(
                codes_get(gid, 'missingValue')
            )

            if param == 'precipitation':

                step_type = str(
                    codes_get(gid, 'stepType')
                )

                start = int(
                    codes_get(gid, 'startStep')
                )

                end = int(
                    codes_get(gid, 'endStep')
                )

            else:
                step_type = ''
                start = None
                end = None

        finally:
            codes_release(gid)

    if (
        uuid != grid_uuid or
        point_count != total_points or
        len(raw) != total_points
    ):
        raise RuntimeError(
            'GRIB-Gitter stimmt nicht zum Tirol-Gitter'
        )

    if short not in PARAMS[param]['short_names']:
        raise RuntimeError(
            'Unerwarteter GRIB-Parameter: ' + short
        )

    selected = np.asarray(
        raw[grid_indices],
        dtype=np.float64
    )

    valid = (
        np.isfinite(selected) &
        (selected != missing_value) &
        (np.abs(selected) < 1e20)
    )

    if param == 'temperature_2m':

        if units != 'K':
            raise RuntimeError(
                'Temperatureinheit: ' + units
            )

        selected[valid] -= 273.15

    elif param == 'precipitation':

        if units not in (
            'kg m**-2',
            'kg m-2',
            'mm'
        ):
            raise RuntimeError(
                'Niederschlagseinheit: ' + units
            )

    elif param == 'cloud_cover':

        if units == '1':
            selected[valid] *= 100

        elif units != '%':
            raise RuntimeError(
                'Bewölkungseinheit: ' + units
            )

        selected[valid] = np.clip(
            selected[valid],
            0,
            100
        )

    return (
        selected,
        valid,
        step_type,
        start,
        end
    )


# ------------------------------------------------------------
# KARTENPROJEKTION
# ------------------------------------------------------------

def mercator_y(degrees):
    radians = np.radians(degrees)

    return np.log(
        np.tan(
            np.pi / 4 + radians / 2
        )
    )


def inverse_mercator(y):
    return np.degrees(
        2 * np.arctan(np.exp(y)) - np.pi / 2
    )


def make_target_grid(w, h):

    x = LON_MIN + (
        np.arange(w) + 0.5
    ) / w * (LON_MAX - LON_MIN)

    y = inverse_mercator(
        mercator_y(LAT_MAX) -
        (np.arange(h) + 0.5) / h *
        (
            mercator_y(LAT_MAX) -
            mercator_y(LAT_MIN)
        )
    )

    return np.meshgrid(x, y)


XX, YY = make_target_grid(
    WIDTH, HEIGHT
)

VX, VY = make_target_grid(
    VALUE_WIDTH, VALUE_HEIGHT
)


# ------------------------------------------------------------
# PUNKTWERTE ALS PNG SPEICHERN
# ------------------------------------------------------------

def encode_values_png(values, valid, filename):

    pixels = np.full(
        values.shape,
        65535,
        dtype=np.uint16
    )

    good = (
        valid &
        np.isfinite(values)
    )

    scaled = np.rint(
        values[good] * 100 + 32768
    )

    if np.any(
        (scaled < 0) |
        (scaled > 65534)
    ):
        raise RuntimeError(
            'Modellwert außerhalb des PNG-Kodierungsbereichs'
        )

    pixels[good] = scaled.astype(
        np.uint16
    )

    rgba = np.empty(
        (*values.shape, 4),
        dtype=np.uint8
    )

    rgba[:, :, 0] = (
        pixels >> 8
    ).astype(np.uint8)

    rgba[:, :, 1] = (
        pixels & 255
    ).astype(np.uint8)

    rgba[:, :, 2] = 0
    rgba[:, :, 3] = 255

    Image.fromarray(
        rgba, 'RGBA'
    ).save(
        filename,
        optimize=True
    )


# ------------------------------------------------------------
# WETTERKARTEN ERZEUGEN
# ------------------------------------------------------------

def create_maps(
    values,
    valid,
    param,
    overlay_path,
    value_path
):

    if np.count_nonzero(valid) < 100:
        raise RuntimeError(
            'Zu wenige gültige Modellpunkte'
        )

    lx = lon[valid]
    ly = lat[valid]
    lv = values[valid]

    tri = mtri.Triangulation(
        lx, ly
    )

    tri_lon = lx[tri.triangles]
    tri_lat = ly[tri.triangles]

    dx = np.diff(
        np.concatenate(
            [
                tri_lon,
                tri_lon[:, :1]
            ],
            axis=1
        ),
        axis=1
    ) * 111.32 * np.cos(
        np.radians(
            (LAT_MIN + LAT_MAX) / 2
        )
    )

    dy = np.diff(
        np.concatenate(
            [
                tri_lat,
                tri_lat[:, :1]
            ],
            axis=1
        ),
        axis=1
    ) * 111.32

    tri.set_mask(
        np.max(
            np.sqrt(dx**2 + dy**2),
            axis=1
        ) > 7
    )

    interp = mtri.LinearTriInterpolator(
        tri, lv
    )

    rendered = np.ma.asarray(
        interp(XX, YY)
    )

    render_valid = (
        ~np.ma.getmaskarray(rendered) &
        np.isfinite(
            rendered.filled(np.nan)
        )
    )

    cfg = PARAMS[param]

    cmap = get_cmap(param)

    norm = Normalize(
        vmin=cfg['limits'][0],
        vmax=cfg['limits'][1],
        clip=True
    )

    rgba = np.uint8(
        cmap(
            norm(
                rendered.filled(
                    cfg['limits'][0]
                )
            )
        ) * 255
    )

    rgba[:, :, 3] = np.where(
        render_valid,
        255,
        0
    ).astype(np.uint8)

    # Niederschlag:
    # 0 mm transparent.
    # Jeder positive Wert wird hellblau bis dunkelblau.

    if param in (
        'precipitation',
        'precipitation_sum'
    ):

        rgba[:, :, 3] = np.where(
            render_valid &
            (rendered.filled(0) > 0.0),
            255,
            0
        ).astype(np.uint8)

        # Kontrollausgabe für GitHub Actions

        n_visible = int(
            np.count_nonzero(
                rgba[:, :, 3]
            )
        )

        n_valid = int(
            np.count_nonzero(
                render_valid
            )
        )

        sample = (
            tuple(
                int(v)
                for v in rgba[
                    rgba[:, :, 3] > 0
                ][0, :3]
            )
            if n_visible
            else None
        )

        print(
            f'    BLAUKONTROLLE {overlay_path.name}: '
            f'{n_visible} sichtbare Niederschlags-Pixel / '
            f'{n_valid} gueltige Pixel; '
            f'Beispiel-RGB={sample}',
            flush=True
        )

    Image.fromarray(
        rgba, 'RGBA'
    ).save(
        overlay_path,
        optimize=True
    )

    del rgba
    del rendered

    sampled = np.ma.asarray(
        interp(VX, VY)
    )

    sampled_good = (
        ~np.ma.getmaskarray(sampled) &
        np.isfinite(
            sampled.filled(np.nan)
        )
    )

    encode_values_png(
        sampled.filled(0),
        sampled_good,
        value_path
    )

    del sampled
    del interp
    del tri

    gc.collect()


# ------------------------------------------------------------
# LEGENDE ERZEUGEN
# ------------------------------------------------------------

def create_legend(param):

    cfg = PARAMS[param]

    lo, hi = cfg['limits']

    fig = plt.figure(
        figsize=(5.8, 1.05),
        dpi=160
    )

    ax = fig.add_axes(
        [0.06, 0.40, 0.88, 0.28]
    )

    cb = ColorbarBase(
        ax,
        cmap=get_cmap(param),
        norm=Normalize(
            vmin=lo,
            vmax=hi
        ),
        orientation='horizontal'
    )

    if param == 'temperature_2m':
        ticks = list(
            range(-5, 26, 5)
        )

    elif param == 'precipitation':
        ticks = [
            0, 1, 2, 5, 10, 15, 20
        ]

    elif param == 'precipitation_sum':
        ticks = [
            0, 5, 10, 20, 30, 40, 50
        ]

    else:
        ticks = [
            0, 20, 40, 60, 80, 100
        ]

    cb.set_ticks(ticks)

    cb.ax.tick_params(
        labelsize=10,
        pad=2
    )

    fig.savefig(
        OUT / cfg['legend'],
        transparent=True,
        bbox_inches='tight',
        pad_inches=0.06
    )

    plt.close(fig)


# ------------------------------------------------------------
# EINEN MODELLLAUF VERARBEITEN
# ------------------------------------------------------------

def process_run(run):

    encoded = quote(
        run,
        safe='-T'
    )

    run_utc = datetime.strptime(
        run,
        '%Y-%m-%dT%H:%M'
    ).replace(
        tzinfo=timezone.utc
    )

    result = {}

    for param, cfg in PARAMS.items():

        if param == 'precipitation_sum':
            continue

        print(
            '\nParameter:',
            cfg['name']
        )

        frames = []

        previous_accum = None
        previous_end = None
        previous_valid = None

        cumulative_sum = None
        cumulative_valid = None

        sum_frames = []

        for hour in HOURS:

            url = (
                f'{DWD_ROOT}'
                f'{cfg["dwd"]}/r/'
                f'{encoded}/s/'
                f'PT{hour:03d}H00M.grib2'
            )

            print(
                f'  +{hour:02d} h:',
                end=' ',
                flush=True
            )

            try:
                size = download(
                    url,
                    GRIB,
                    20
                )

                (
                    values,
                    valid,
                    step_type,
                    start,
                    end
                ) = read_grib(
                    GRIB,
                    param
                )

            finally:
                GRIB.unlink(
                    missing_ok=True
                )

            if param == 'precipitation':

                if (
                    step_type != 'accum' or
                    end != hour
                ):
                    raise RuntimeError(
                        'Unerwartetes Niederschlagsintervall: '
                        f'{step_type} {start}-{end} h'
                    )

                if start == hour - 1:

                    hourly = values.copy()

                elif start == 0:

                    if hour == 1:
                        hourly = values.copy()

                    else:

                        if (
                            previous_accum is None or
                            previous_end != hour - 1 or
                            previous_valid is None
                        ):
                            raise RuntimeError(
                                'Vorherige Niederschlagsakkumulation fehlt'
                            )

                        valid = (
                            valid &
                            previous_valid
                        )

                        hourly = (
                            values -
                            previous_accum
                        )

                else:
                    raise RuntimeError(
                        'Unbekanntes Niederschlagsintervall '
                        f'{start}-{end} h'
                    )

                raw_precip = values.copy()
                raw_valid = valid.copy()

                # Vorherige Akkumulation erst nach
                # der Differenzberechnung aktualisieren.

                if start == 0:

                    previous_accum = values.copy()
                    previous_end = end
                    previous_valid = valid.copy()

                else:

                    previous_accum = None
                    previous_end = None
                    previous_valid = None

                if np.any(
                    hourly[valid] < -0.01
                ):
                    raise RuntimeError(
                        'Negative Niederschlagsdifferenz'
                    )

                values = np.maximum(
                    hourly,
                    0
                )

                # Niederschlagssumme seit Modellbeginn

                if start == 0:

                    cumulative_sum = np.maximum(
                        raw_precip,
                        0
                    )

                    cumulative_valid = (
                        raw_valid.copy()
                    )

                else:

                    if cumulative_sum is None:

                        cumulative_sum = (
                            values.copy()
                        )

                        cumulative_valid = (
                            valid.copy()
                        )

                    else:

                        cumulative_sum = (
                            cumulative_sum +
                            values
                        )

                        cumulative_valid = (
                            cumulative_valid &
                            valid
                        )

                print(
                    f'Intervall {start}-{end} h |',
                    end=' '
                )

            # Wetterbild der Prognosestunde

            png = (
                f'icon_d2_ruc_'
                f'{param}_{hour:03d}.png'
            )

            value_png = (
                f'icon_d2_ruc_'
                f'{param}_{hour:03d}_werte.png'
            )

            create_maps(
                values,
                valid,
                param,
                OUT / png,
                OUT / value_png
            )

            frames.append({
                'forecast_hour': hour,
                'valid_time_utc': (
                    run_utc +
                    timedelta(hours=hour)
                ).isoformat(),
                'image': png,
                'values_image': value_png
            })

            # Zusätzlich Niederschlagssumme erzeugen

            if param == 'precipitation':

                sum_png = (
                    'icon_d2_ruc_'
                    f'precipitation_sum_{hour:03d}.png'
                )

                sum_value_png = (
                    'icon_d2_ruc_'
                    f'precipitation_sum_{hour:03d}_werte.png'
                )

                create_maps(
                    cumulative_sum,
                    cumulative_valid,
                    'precipitation_sum',
                    OUT / sum_png,
                    OUT / sum_value_png
                )

                sum_frames.append({
                    'forecast_hour': hour,
                    'valid_time_utc': (
                        run_utc +
                        timedelta(hours=hour)
                    ).isoformat(),
                    'image': sum_png,
                    'values_image': sum_value_png
                })

            print(
                f'OK | {size / 1024:.1f} KB'
            )

        result[param] = {
            'name': cfg['name'],
            'unit': cfg['unit'],
            'legend': cfg['legend'],
            'frames': frames
        }

        if param == 'precipitation':

            sum_cfg = PARAMS[
                'precipitation_sum'
            ]

            result['precipitation_sum'] = {
                'name': sum_cfg['name'],
                'unit': sum_cfg['unit'],
                'legend': sum_cfg['legend'],
                'frames': sum_frames
            }

    return result, run_utc


# ------------------------------------------------------------
# NEUESTEN VOLLSTÄNDIGEN MODELLLAUF SUCHEN
# ------------------------------------------------------------

selected_data = None
selected_run = None

for run in runs[:8]:

    print(
        '\nPrüfe Modelllauf:',
        run
    )

    for old in OUT.iterdir():
        if old.is_file():
            old.unlink()

    try:
        selected_data, selected_run = process_run(
            run
        )

        break

    except Exception as e:

        print(
            'Modelllauf nicht vollständig:',
            type(e).__name__,
            str(e)
        )

if selected_data is None:
    raise RuntimeError(
        'Kein vollständiger Modelllauf gefunden.'
    )


# ------------------------------------------------------------
# LEGENDE ALLER PARAMETER ERZEUGEN
# ------------------------------------------------------------

for param in PARAMS:
    create_legend(param)


# ------------------------------------------------------------
# METADATEN ERZEUGEN
# ------------------------------------------------------------

first = selected_data[
    'temperature_2m'
]['frames'][0]

metadata = {
    'model': 'ICON-D2-RUC',
    'provider': 'DWD',

    'generated_at_utc': datetime.now(
        timezone.utc
    ).isoformat(),

    'model_run_utc': selected_run.isoformat(),

    'bounds': [
        [LAT_MIN, LON_MIN],
        [LAT_MAX, LON_MAX]
    ],

    'center': [
        47.2692,
        11.4041
    ],

    'projection': 'EPSG:3857',

    'width': WIDTH,
    'height': HEIGHT,

    'values_encoding': {
        'kind': 'png-rg16-offset',
        'scale': 100,
        'offset': 32768,
        'missing': 65535,
        'width': VALUE_WIDTH,
        'height': VALUE_HEIGHT
    },

    'parameters': selected_data,

    'parameter': 'temperature_2m',

    'image': first['image'],

    'legend': PARAMS[
        'temperature_2m'
    ]['legend'],

    'valid_time_utc': first['valid_time_utc'],

    'grid_uuid': grid_uuid
}

JSON_FILE = (
    OUT /
    'icon_d2_ruc_tirol_metadata.json'
)

JSON_FILE.write_text(
    json.dumps(
        metadata,
        ensure_ascii=False,
        indent=2
    ),
    encoding='utf-8'
)


# ------------------------------------------------------------
# ALLE ERGEBNISDATEIEN PRÜFEN
# ------------------------------------------------------------

for param in PARAMS:

    frames = metadata[
        'parameters'
    ][param]['frames']

    if len(frames) != 12:
        raise RuntimeError(
            'Fehlende Stunden bei ' + param
        )

    for frame in frames:

        for field in (
            'image',
            'values_image'
        ):

            path = OUT / frame[field]

            if not path.exists():
                raise RuntimeError(
                    'Fehlende Datei: ' + str(path)
                )

            with Image.open(path) as im:
                im.verify()

    with Image.open(
        OUT / PARAMS[param]['legend']
    ) as im:
        im.verify()

    print(
        param +
        ': 12 Wetterbilder und 12 Wertebilder geprüft'
    )


# ------------------------------------------------------------
# SPEICHERBEDARF
# ------------------------------------------------------------

size = sum(
    p.stat().st_size
    for p in OUT.iterdir()
    if p.is_file()
)

print(
    '\nModelllauf UTC:',
    selected_run.isoformat()
)

print(
    'Gesamtgröße der Wetterdateien:',
    round(size / 1024**2, 2),
    'MB'
)

print(
    'Anzahl Ausgabedateien:',
    len(list(OUT.iterdir()))
)


# ------------------------------------------------------------
# ABSCHLUSS
# ------------------------------------------------------------

for param in PARAMS:

    print(
        '\nVorschau',
        PARAMS[param]['name'],
        '+6 h'
    )

shutil.rmtree(
    TMP,
    ignore_errors=True
)

zip_path = shutil.make_archive(
    'ICON_D2_RUC_Tirol_4_Parameter_mit_Werten',
    'zip',
    root_dir=OUT
)

print(
    '\nAlle 48 Karten und 48 Wertebilder '
    'erstellt und geprüft.'
)

print(
    'ZIP:',
    zip_path
)

print(
    'Die Wetterdaten können jetzt '
    'über GitHub Pages veröffentlicht werden.'
)
