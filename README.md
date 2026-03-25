# Taller ML - MIAX (Parte 1)

Este proyecto contiene el notebook **`src/MIAX_ML_Workshop_Part1_windows.ipynb`**, centrado en el preprocesado de datos financieros y la construcción de variables para clustering y selección de fondos.

## Resumen del notebook

El cuaderno transforma series históricas de NAV por fondo en una base analítica robusta y lista para modelado.

1. **Contexto y objetivo**
   - Universo de ~25.000 fondos con datos diarios (2016-01-05 a 2021-07-16).
   - Tesis: identificar fondos alineados con el comportamiento de **Asia Pacific ex Japan**.

2. **Carga y auditoría de datos (`navs.pickle`)**
   - El origen es un diccionario `allfunds_id -> DataFrame temporal`.
   - Se revisa calidad global: longitud de series, fechas, nulos y saltos extremos en retornos.

3. **Limpieza reproducible**
   - Filtros principales aplicados:
     - `n_obs >= 252` (mínimo de historial útil).
     - `max_abs_ret <= 0.50` (control de saltos anómalos).

4. **Integración de factores de riesgo (Fama-French + Momentum)**
   - Carga de `Mkt-RF`, `SMB`, `HML`, `RF` y `WML`.
   - Parseo robusto de fechas y unificación por intersección temporal.
   - Conversión de porcentaje diario a escala decimal.

5. **Feature engineering por fondo**
   - Métricas de retorno/riesgo: retorno anualizado, volatilidad, Sharpe, Sortino.
   - Riesgo de pérdidas: máximo drawdown.
   - Forma de distribución: asimetría y curtosis.
   - Exposiciones factoriales mediante OLS: `beta_mkt`, `beta_smb`, `beta_hml`, `beta_wml`.
   - Indicadores semánticos del nombre (flags de Asia/Japan/Pacific).

6. **Matriz final para clustering (`X_clust`)**
   - Selección de variables financieras finales.
   - Tratamiento de infinitos y nulos (imputación por mediana).
   - Estandarización con `StandardScaler`.

7. **Benchmark y regla de selección**
   - Construcción de benchmark proxy Asia ex Japan con `R_m = (Mkt-RF) + RF`.
   - Ranking con criterio cuantitativo explícito basado en exposición (`beta_mkt`), calidad retorno-riesgo y control de drawdown.

8. **Persistencia y reproducibilidad**
   - Exportación de datasets intermedios/finales.
   - Guardado del escalador y metadatos para reutilizar la misma transformación en Parte 2.

## Estructura

- `src/MIAX_ML_Workshop_Part1_windows.ipynb`: notebook principal.
- `dataset/`: datos de entrada y artefactos generados (`.csv`, `.json`, `.pkl`, `.pickle`).
- `requirements.txt`: dependencias de Python necesarias para ejecutar el notebook.

## Artefactos generados por el notebook

- `dataset/features_fondos_part1.csv`: tabla de características por fondo.
- `dataset/X_clust_part1.csv`: matriz final estandarizada para clustering.
- `dataset/scaler_x_clust_part1.pkl`: objeto `StandardScaler` persistido.
- `dataset/X_clust_part1_metadata.json`: metadatos de estandarización y uso.

## Requisitos previos

- Python 3.10 o superior
- `pip` actualizado

## Instalación

Desde la raíz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecución del notebook

```powershell
jupyter notebook
```

Después, abre:

`src/MIAX_ML_Workshop_Part1_windows.ipynb`

## Notas

- El notebook usa rutas relativas al proyecto (por ejemplo `dataset/...`), por lo que conviene lanzarlo desde la raíz.
- `X_clust_part1.csv` ya está estandarizado; no volver a escalar en la Parte 2.
