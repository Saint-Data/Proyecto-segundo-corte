"""
utils.py — Funciones compartidas por los 4 notebooks del proyecto UrbanMove.
Proyecto Retador — ML No Supervisado — Universidad de La Sabana 2026-II

Contiene:
  * haversine_km(): distancia geodésica (la Tierra NO es plana; nunca usamos
    distancia euclidiana directa sobre lat/lon).
  * limpiar(): filtros físicos de calidad sobre el train.csv de Kaggle.
  * construir_muestra(): muestra aleatoria simple de 50 000 viajes (semilla 42).
  * cargar_muestra(): carga la muestra ya construida (data/muestra_urbanmove.csv).
  * asignar_zona(): grid geográfico de ~1.7 km x 1.7 km para análisis por zona.
  * a_km(): proyección equirectangular local a km (para K-means / GMM).
"""
import os
import json
import numpy as np
import pandas as pd

SEMILLA = 42
N_MUESTRA = 50_000
RUTA_TRAIN = os.path.join("data", "train.csv")          # descargado de Kaggle
RUTA_MUESTRA = os.path.join("data", "muestra_urbanmove.csv")
RUTA_RESULTADOS = "resultados"

# Caja geográfica de NYC (5 boroughs + aeropuertos JFK / LGA / EWR cercanos)
LON_MIN, LON_MAX = -74.26, -73.70
LAT_MIN, LAT_MAX = 40.49, 40.92

# Punto de referencia para proyección local (Midtown Manhattan)
LAT0, LON0 = 40.75, -73.98
R_TIERRA_KM = 6371.0088

# Grid de zonas: 0.015° de latitud (~1.67 km) x 0.02° de longitud (~1.69 km a 40.75°N)
DLAT, DLON = 0.015, 0.02
UMBRAL_ZONA = 0.005     # celdas con < 0.5 % de los pickups se agrupan en "Otras"

FRANJAS = {  # franja horaria -> horas incluidas
    "Madrugada (0-5h)": range(0, 6),
    "Pico AM (6-9h)": range(6, 10),
    "Valle (10-15h)": range(10, 16),
    "Pico PM (16-19h)": range(16, 20),
    "Noche (20-23h)": range(20, 24),
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia geodésica sobre la esfera (km). Entradas en grados."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R_TIERRA_KM * np.arcsin(np.sqrt(a))


def a_km(lat, lon):
    """Proyección equirectangular local (x hacia el este, y hacia el norte) en km.
    Error < 0.5 % dentro de NYC, suficiente para clustering con distancia euclidiana."""
    x = np.radians(lon - LON0) * R_TIERRA_KM * np.cos(np.radians(LAT0))
    y = np.radians(lat - LAT0) * R_TIERRA_KM
    return x, y


def franja(hora):
    for nombre, horas in FRANJAS.items():
        if hora in horas:
            return nombre
    raise ValueError(hora)


def agregar_variables(df):
    """Variables derivadas: temporales, distancia Haversine, velocidad, coordenadas en km."""
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hora"] = df["pickup_datetime"].dt.hour
    df["dia_semana"] = df["pickup_datetime"].dt.dayofweek          # 0 = lunes
    df["mes"] = df["pickup_datetime"].dt.month
    df["fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    df["franja"] = df["hora"].map(franja)
    df["dist_km"] = haversine_km(df.pickup_latitude, df.pickup_longitude,
                                 df.dropoff_latitude, df.dropoff_longitude)
    df["duracion_min"] = df["trip_duration"] / 60.0
    df["vel_kmh"] = df["dist_km"] / (df["trip_duration"] / 3600.0)
    df["px"], df["py"] = a_km(df.pickup_latitude, df.pickup_longitude)
    df["dx"], df["dy"] = a_km(df.dropoff_latitude, df.dropoff_longitude)
    return df


def limpiar(df, verbose=True):
    """Filtros físicos de calidad. Devuelve (df_limpio, tabla_de_filtros)."""
    pasos = []
    n0 = len(df)
    pasos.append(("Registros originales (train.csv Kaggle)", n0))

    def paso(nombre, mascara):
        nonlocal df
        df = df[mascara]
        pasos.append((nombre, len(df)))

    paso("Duración entre 1 min y 3 h", df.trip_duration.between(60, 3 * 3600))
    dentro = (df.pickup_longitude.between(LON_MIN, LON_MAX) & df.pickup_latitude.between(LAT_MIN, LAT_MAX)
              & df.dropoff_longitude.between(LON_MIN, LON_MAX) & df.dropoff_latitude.between(LAT_MIN, LAT_MAX))
    paso("Origen y destino dentro de la caja de NYC", dentro)
    paso("Pasajeros entre 1 y 6", df.passenger_count.between(1, 6))
    df = agregar_variables(df)
    paso("Distancia Haversine >= 0.1 km", df.dist_km >= 0.1)
    paso("Velocidad promedio <= 100 km/h (físicamente posible)", df.vel_kmh <= 100)
    tabla = pd.DataFrame(pasos, columns=["Filtro", "Registros restantes"])
    tabla["Eliminados"] = (-tabla["Registros restantes"].diff()).fillna(0).astype(int)
    tabla["% del original"] = 100 * tabla["Registros restantes"] / n0
    if verbose:
        print(tabla.to_string(index=False))
    return df, tabla


# Etiquetas aproximadas (barrio de referencia) del centro de cada celda del grid.
# Son solo una ayuda de lectura; el análisis usa la celda, no el barrio.
ETIQUETAS = {
    (40.753, -73.990): "Midtown Oeste/Penn Station", (40.738, -73.990): "Chelsea/Flatiron",
    (40.753, -73.970): "Midtown Este/Grand Central", (40.768, -73.970): "Upper East Side Sur",
    (40.768, -73.990): "Columbus Circle/Hell's Kitchen", (40.723, -73.990): "East Village/LES",
    (40.782, -73.950): "Upper East Side Norte", (40.738, -74.010): "West Village",
    (40.768, -73.950): "Lenox Hill (UES)", (40.782, -73.970): "Central Park/UWS",
    (40.723, -74.010): "SoHo/Tribeca", (40.708, -74.010): "Financial District",
    (40.768, -73.870): "Aeropuerto LaGuardia", (40.797, -73.970): "Upper West Side Norte",
    (40.782, -73.990): "Upper West Side", (40.648, -73.790): "Aeropuerto JFK",
    (40.738, -73.970): "Gramercy/Kips Bay", (40.753, -74.010): "Hudson Yards/Chelsea Oeste",
    (40.797, -73.950): "East Harlem",
}


def construir_zonas(df, umbral=UMBRAL_ZONA):
    """Asigna cada pickup/dropoff a una celda del grid. Celdas con < umbral de los pickups
    se agrupan en la zona 'Otras'. Las zonas se nombran por el centro de la celda."""
    def celda(lat, lon):
        i = np.floor((lat - LAT_MIN) / DLAT).astype(int)
        j = np.floor((lon - LON_MIN) / DLON).astype(int)
        return i * 1000 + j

    cp = celda(df.pickup_latitude.values, df.pickup_longitude.values)
    cd = celda(df.dropoff_latitude.values, df.dropoff_longitude.values)
    frec = pd.Series(cp).value_counts(normalize=True)
    validas = frec[frec >= umbral].index
    orden = {c: k for k, c in enumerate(sorted(validas, key=lambda c: -frec[c]))}
    nombre = {}
    for c, k in orden.items():
        i, j = divmod(c, 1000)
        lat_c = LAT_MIN + (i + 0.5) * DLAT
        lon_c = LON_MIN + (j + 0.5) * DLON
        etiqueta = ETIQUETAS.get((round(lat_c, 3), round(lon_c, 3)), f"({lat_c:.3f},{lon_c:.3f})")
        nombre[c] = f"Z{k:02d} {etiqueta}"
    df = df.copy()
    df["zona_pickup"] = [nombre.get(c, "Otras") for c in cp]
    df["zona_dropoff"] = [nombre.get(c, "Otras") for c in cd]
    return df


def construir_muestra(ruta_train=RUTA_TRAIN, n=N_MUESTRA, semilla=SEMILLA, guardar=True):
    crudo = pd.read_csv(ruta_train)
    limpio, tabla = limpiar(crudo)
    muestra = limpio.sample(n=n, random_state=semilla).reset_index(drop=True)
    muestra = construir_zonas(muestra)
    if guardar:
        os.makedirs("data", exist_ok=True)
        muestra.to_csv(RUTA_MUESTRA, index=False)
    return crudo, limpio, muestra, tabla


def cargar_muestra():
    """Carga la muestra de 50 000 viajes. Si no existe, la construye desde data/train.csv."""
    if not os.path.exists(RUTA_MUESTRA):
        print("Muestra no encontrada: construyendo desde", RUTA_TRAIN)
        construir_muestra()
    df = pd.read_csv(RUTA_MUESTRA, parse_dates=["pickup_datetime"])
    return df


def guardar_resultados(nombre, dic):
    os.makedirs(RUTA_RESULTADOS, exist_ok=True)
    with open(os.path.join(RUTA_RESULTADOS, nombre), "w", encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False, indent=2, default=float)


def leer_resultados(nombre):
    with open(os.path.join(RUTA_RESULTADOS, nombre), encoding="utf-8") as f:
        return json.load(f)
