# ============================================================================
# ARCHIVO: data_prep.py
# ============================================================================
import yfinance as yf
import pandas as pd
import numpy as np

def descargar_y_preparar_datos(ticker_forex="EURJPY=X"):
    """
    Descarga un año de datos horarios, realiza la partición semanal estricta
    para evitar Data Leakage y exporta los archivos CSV correspondientes.
    """
    print(f"=== INICIANDO DESCARGA DE DATOS PARA {ticker_forex} ===")
    
    # 1. Descarga de datos
    df_raw = yf.download(ticker_forex, period="1y", interval="1h")
    
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
    
    df_raw.reset_index(inplace=True)
    df_raw.columns = [col.lower() for col in df_raw.columns]
    df_raw.rename(columns={'datetime': 'time', 'date': 'time'}, inplace=True)
    
    print(f"-> Registros descargados de Yahoo Finance: {len(df_raw)}")
    
    # 2. Partición Cronológica Estricta (Alineación de Lunes)
    df_raw['time'] = pd.to_datetime(df_raw['time']).dt.tz_localize(None)
    df_raw = df_raw.sort_values('time').reset_index(drop=True)
    
    ultima_fecha = df_raw['time'].max()
    dias_desde_lunes = ultima_fecha.weekday()
    lunes_actual = (ultima_fecha - pd.Timedelta(days=dias_desde_lunes)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    df_historico = df_raw[df_raw['time'] < lunes_actual].copy()
    df_semana_actual = df_raw[df_raw['time'] >= lunes_actual].copy()
    
    primera_fecha_hist = df_historico['time'].min()
    dias_para_primer_lunes = (7 - primera_fecha_hist.weekday()) % 7
    primer_lunes_hist = (primera_fecha_hist + pd.Timedelta(days=dias_para_primer_lunes)).replace(hour=0, minute=0, second=0, microsecond=0)
    df_historico = df_historico[df_historico['time'] >= primer_lunes_hist].copy()
    
    print(f"-> Datos de Entrenamiento (Semanas Completas): {len(df_historico)} velas")
    print(f"-> Datos de Forward Test (Semana Actual): {len(df_semana_actual)} velas")
    
    # 3. Exportación física de los Datasets
    # Extraemos el prefijo del ticker (ej: "EURJPY=X" -> "EURJPY")
    symb = ticker_forex.split('=')[0] if '=' in ticker_forex else ticker_forex
    
    archivo_historico = f"{symb}_H1_Hist_Semanas_Completas.csv"
    archivo_prueba = f"{symb}_H1_Semana_Actual_Prueba.csv"
    archivo_base = f"{symb}_D1_ultimo_ano.csv"
    
    df_historico.to_csv(archivo_historico, index=False)
    df_semana_actual.to_csv(archivo_prueba, index=False)
    df_raw.to_csv(archivo_base, index=False)
    
    print("=== PROCESO DE ADQUISICIÓN COMPLETADO EXITOSAMENTE ===")
    print(f"Archivos listos en el disco local: \n1. {archivo_historico} \n2. {archivo_prueba} \n3. {archivo_base}")
    
    # Retornamos los DataFrames para que el usuario pueda usarlos inmediatamente en memoria si lo desea
    return df_historico, df_semana_actual, df_raw
