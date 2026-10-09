            finally:
                GRIB.unlink(missing_ok=True)
            if param=='precipitation':
                if step_type!='accum' or end!=hour:
                    raise RuntimeError(f'Unerwartetes Niederschlagsintervall: {step_type} {start}-{end} h')
                if start==hour-1:
                    hourly=values.copy()
                elif start==0:
                    if hour==1:
                        hourly=values.copy()
                    else:
                        if previous_accum is None or previous_end!=hour-1 or previous_valid is None:
                            raise RuntimeError('Vorherige Niederschlagsakkumulation fehlt')
                        valid=valid&previous_valid
                        hourly=values-previous_accum
                else:
                    raise RuntimeError(f'Unbekanntes Niederschlagsintervall {start}-{end} h')
                raw_precip=values.copy()
                raw_valid=valid.copy()
                # Erst NACH der Berechnung aktualisieren, bei +1 h ebenfalls.
                if start==0:
                    previous_accum=values.copy()
                    previous_end=end
                    previous_valid=valid.copy()
                else:
                    previous_accum=previous_end=previous_valid=None
                if np.any(hourly[valid]<-0.01):
                    raise RuntimeError('Negative Niederschlagsdifferenz; Intervalle prüfen')
                values=np.maximum(hourly,0)
                # Niederschlagssumme vom Modellbeginn bis zur Vorhersagestunde.
                # Bei 0-h-Akkumulation direkt den Gesamtwert übernehmen;
                # bei Stundenintervallen die Einzelstunden aufsummieren.
                if start==0:
                    cumulative_sum=np.maximum(raw_precip,0)
                    cumulative_valid=raw_valid.copy()
                else:
                    if cumulative_sum is None:
                        cumulative_sum=values.copy()
                        cumulative_valid=valid.copy()
                    else:
                        cumulative_sum=cumulative_sum+values
                        cumulative_valid=np.logical_and(cumulative_valid,valid)
                print(f'Intervall {start}-{end} h |',end=' ')
            png=f'icon_d2_ruc_{param}_{hour:03d}.png'
            value_png=f'icon_d2_ruc_{param}_{hour:03d}_werte.png'
            create_maps(values,valid,param,OUT/png,OUT/value_png)
            frames.append({'forecast_hour':hour,'valid_time_utc':(run_utc+timedelta(hours=hour)).isoformat(),'image':png,'values_image':value_png})
            if param=='precipitation':
                sum_png=f'icon_d2_ruc_precipitation_sum_{hour:03d}.png'
                sum_value_png=f'icon_d2_ruc_precipitation_sum_{hour:03d}_werte.png'
                create_maps(cumulative_sum,cumulative_valid,'precipitation_sum',OUT/sum_png,OUT/sum_value_png)
                sum_frames.append({'forecast_hour':hour,'valid_time_utc':(run_utc+timedelta(hours=hour)).isoformat(),'image':sum_png,'values_image':sum_value_png})
            print(f'OK | {size/1024:.1f} KB')
        result[param]={'name':cfg['name'],'unit':cfg['unit'],'legend':cfg['legend'],'frames':frames}
        if param=='precipitation':
            sum_cfg=PARAMS['precipitation_sum']
            result['precipitation_sum']={'name':sum_cfg['name'],'unit':sum_cfg['unit'],'legend':sum_cfg['legend'],'frames':sum_frames}
    return result,run_utc

selected_data=selected_run=None
for run in runs[:8]:
    print('\nPrüfe Modelllauf:',run)
    for old in OUT.iterdir():
        if old.is_file(): old.unlink()
    try:
        selected_data,selected_run=process_run(run)
        break
    except Exception as e:
        print('Modelllauf nicht vollständig:',type(e).__name__,str(e))
if selected_data is None:
    raise RuntimeError('Kein vollständiger Modelllauf gefunden. Bitte die Ausgabe senden, noch nichts hochladen.')
for param in PARAMS: create_legend(param)

first=selected_data['temperature_2m']['frames'][0]
metadata={
 'model':'ICON-D2-RUC','provider':'DWD','generated_at_utc':datetime.now(timezone.utc).isoformat(),
 'model_run_utc':selected_run.isoformat(),
 'bounds':[[LAT_MIN,LON_MIN],[LAT_MAX,LON_MAX]],
 'center':[47.2692,11.4041],
 'projection':'EPSG:3857','width':WIDTH,'height':HEIGHT,
 'values_encoding':{'kind':'png-rg16-offset','scale':100,'offset':32768,'missing':65535,'width':VALUE_WIDTH,'height':VALUE_HEIGHT},
 'parameters':selected_data,
 'parameter':'temperature_2m','image':first['image'],'legend':PARAMS['temperature_2m']['legend'],
 'valid_time_utc':first['valid_time_utc'],'grid_uuid':grid_uuid
}
JSON_FILE=OUT/'icon_d2_ruc_tirol_metadata.json'
JSON_FILE.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
for param in PARAMS:
    frames=metadata['parameters'][param]['frames']
    if len(frames)!=12: raise RuntimeError('Fehlende Stunden bei '+param)
    for frame in frames:
        for field in ('image','values_image'):
            path=OUT/frame[field]
            if not path.exists(): raise RuntimeError('Fehlende Datei: '+str(path))
            with Image.open(path) as im: im.verify()
    with Image.open(OUT/PARAMS[param]['legend']) as im: im.verify()
    print(param+': 12 Wetterbilder und 12 Wertebilder geprüft')
size=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
print('\nModelllauf UTC:',selected_run.isoformat())
print('Gesamtgröße auf World4You nach Upload:',round(size/1024**2,2),'MB')
print('Anzahl Ausgabedateien:',len(list(OUT.iterdir())))
for param in PARAMS:
    print('\nVorschau',PARAMS[param]['name'],'+6 h')
shutil.rmtree(TMP,ignore_errors=True)
zip_path=shutil.make_archive('ICON_D2_RUC_Tirol_4_Parameter_mit_Werten','zip',root_dir=OUT)
print('\n✅ Alle 48 Karten und 48 Wertebilder erstellt und geprüft.')
print('ZIP:',zip_path)
print('Die ZIP-Datei wird durch GitHub Actions als Test-Artefakt gespeichert.')
