# Patrones Invisibles: Información, Incertidumbre y Movilidad Urbana

Proyecto Retador — **ML No Supervisado**, Universidad de La Sabana, 2026-II (Prof. Eduardo Carrillo B.)

Consultoría analítica para **UrbanMove** sobre el dataset [NYC Taxi Trip Duration (Kaggle)](https://www.kaggle.com/c/nyc-taxi-trip-duration/data).

## Equipo

| Integrante | Código | Rol | Notebook |
|---|---|---|---|
| Rafael José Correa Redondo | [código] | Rol 1 — Analista de Incertidumbre e Información | `Rol1_Incertidumbre_Informacion.ipynb` |
| Santiago Garzón Galvis | [código] | Rol 2 — Arquitecto de Segmentación por Partición | `Rol2_Segmentacion_Particion.ipynb` |
| Nicolás David Cañón | [código] | Rol 3 — Modelador de Mezclas y Variables Latentes | `Rol3_Mezclas_Variables_Latentes.ipynb` |
| Gabriel Pulido| [código] | Rol 4 — Estratega de Negocio y Selección de Datos | `Rol4_Negocio_Seleccion_Datos.ipynb` |

## Estructura

```
├── utils.py                               # funciones compartidas: limpieza, Haversine, zonas, muestra
├── Rol1_Incertidumbre_Informacion.ipynb   # entropía, entropía condicional, MI, KL, entropía cruzada, Jensen, log-sum-exp
├── Rol2_Segmentacion_Particion.ipynb      # K-means, jerárquico, DBSCAN (haversine), métricas, ruido
├── Rol3_Mezclas_Variables_Latentes.ipynb  # GMM con EM implementado desde cero, inicializaciones, convergencia
├── Rol4_Negocio_Seleccion_Datos.ipynb     # selección de datos, representatividad, integración y recomendaciones
├── data/muestra_urbanmove.csv             # muestra aleatoria de 50 000 viajes (semilla 42)
├── resultados/                            # métricas de cada rol en JSON (las lee el Rol 4)
└── figuras/                               # figuras exportadas por los notebooks
```

## Cómo reproducir

```bash
pip install -r requirements.txt
```

1. (Opcional) Descargar `train.csv` de Kaggle y ponerlo en `data/`. Solo se necesita para regenerar la muestra y para la comparación muestra-vs-población del Rol 4; la muestra ya está en el repositorio.
2. Ejecutar en orden: `Rol1` → `Rol2` → `Rol3` → `Rol4` (el Rol 3 usa las etiquetas de ruido del Rol 2 y el Rol 4 lee los resultados de los tres).

Todos los notebooks se entregan **ejecutados** y usan semilla fija (42), por lo que los números son reproducibles.

## Decisiones metodológicas clave

- **Muestra:** 50 000 viajes aleatorios de 1 439 310 limpios (filtros físicos: duración 1 min–3 h, dentro de NYC, 1–6 pasajeros, distancia ≥ 0.1 km, velocidad ≤ 100 km/h). Verificada con KL muestra‖población ≤ 0.0004 bits.
- **Distancias:** Haversine para la distancia de viaje; proyección local en km para K-means/jerárquico; métrica haversine en DBSCAN. Nunca distancia euclidiana sobre lat/lon.
- **Zonas:** grid de ~1.7 km × 1.7 km, 19 celdas con ≥ 0.5 % de la demanda + `Otras` (20 zonas).
- **EM:** implementado en NumPy (paso E con log-sum-exp, paso M cerrado), verificado contra `sklearn`.
