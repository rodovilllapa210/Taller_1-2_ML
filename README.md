# Taller ML - MIAX

Este repositorio recoge la práctica completa del taller de **Machine Learning supervisado y no supervisado aplicado a fondos de inversión**, organizada en una Parte 1 de construcción de variables, una Parte 2 de reducción de dimensionalidad y clustering, y dos cuadernos adicionales de validación final frente al ETF **AAXJ**.

El objetivo general es identificar y analizar fondos alineados con la tesis **Asia Pacific ex Japan**, partiendo de series históricas de NAV y de factores de riesgo diarios.

Mi intención hubiera sido hacer una práctica paralela a partir de las correcciones de la parte 1 del día 26/03 pero no me ha dado tiempo. Así que completo la práctica a partir de mi parte 1.

## Estructura del proyecto

- `src/MIAX_ML_Workshop_Part1_windows.ipynb`: notebook principal de la Parte 1. Construye las features por fondo, prepara la matriz para clustering y exporta los artefactos base del pipeline.
- `src/MIAX_ML_Workshop_Part2.ipynb`: notebook de la Parte 2. Aplica PCA, detección de outliers, clustering y exporta los resultados finales del análisis no supervisado.
- `src/MIAX_AAXJ_Final_Test.ipynb`: notebook de validación final. Genera las features del ETF AAXJ en la misma ventana temporal y con la misma lógica de la Parte 1.
- `src/MIAX_AAXJ_Cluster_Comparison.ipynb`: notebook comparativo entre AAXJ y la cartera sintética del clúster candidato Asia.
- `src/aaxj_feature_builder.py`: script reutilizable que descarga la serie oficial de AAXJ desde iShares, la cruza con factores y genera `dataset/aaxj_features_final_test.csv`.
- `dataset/`: datos de entrada y ficheros de salida generados por los notebooks y el script auxiliar.
- `requirements.txt`: dependencias necesarias para ejecutar los notebooks y el script.

## Flujo de trabajo

### Parte 1: preprocesado y feature engineering

El notebook `src/MIAX_ML_Workshop_Part1_windows.ipynb` realiza:

1. Carga y auditoría del universo de fondos desde `dataset/navs.pickle`.
2. Limpieza reproducible de series según cobertura mínima y control de saltos anómalos.
3. Integración de factores Fama-French y momentum para Asia Pacific ex Japan.
4. Construcción de variables de retorno, riesgo, drawdown, forma de distribución y betas factoriales.
5. Generación de la matriz final `X_clust` para clustering.
6. Persistencia del `StandardScaler` y de metadatos para reutilizar exactamente la misma transformación en la Parte 2.
7. Exportación de una selección de fondos alineados con la tesis Asia.

### Parte 2: reducción de dimensionalidad y clustering

El notebook `src/MIAX_ML_Workshop_Part2.ipynb` toma como entrada los artefactos de la Parte 1 y realiza:

1. Carga de `features_fondos_part1.csv`, `X_clust_part1.csv` y `X_clust_part1_metadata.json`.
2. Revisión de calidad y separación auditable de outliers.
3. Reducción de dimensionalidad mediante PCA.
4. Segmentación con métodos de clustering, incluyendo K-Means y clustering jerárquico aglomerativo.
5. Exportación del universo listo para visualización externa y de la selección final de fondos candidatos.
6. Validación final proyectando el ETF AAXJ con las mismas variables y el mismo escalado guardado en la Parte 1.

### Validación final con AAXJ

Los cuadernos `src/MIAX_AAXJ_Final_Test.ipynb` y `src/MIAX_AAXJ_Cluster_Comparison.ipynb` completan la validación:

- `MIAX_AAXJ_Final_Test.ipynb` genera las features del ETF **iShares MSCI All Country Asia ex Japan ETF (AAXJ)** en la ventana **2016-01-05 a 2021-07-16**.
- `MIAX_AAXJ_Cluster_Comparison.ipynb` construye una cartera sintética equiponderada con los fondos candidatos exportados en la Parte 2 y compara su evolución frente a AAXJ.

## Datos de entrada

- `dataset/navs.pickle`: universo de fondos con series temporales de NAV.
- `dataset/ff_asia_pacific_3factors_daily.csv`: factores diarios `Mkt-RF`, `SMB`, `HML` y `RF`.
- `dataset/mom_asia_pacific_daily.csv`: factor diario de momentum `WML`.

## Artefactos generados

### Salidas de la Parte 1

- `dataset/features_fondos_part1.csv`: tabla de características por fondo.
- `dataset/X_clust_part1.csv`: matriz final estandarizada para clustering.
- `dataset/scaler_x_clust_part1.pkl`: objeto `StandardScaler` persistido.
- `dataset/X_clust_part1_metadata.json`: metadatos de estandarización y uso.
- `dataset/fondos_alineados_tesis_asia_part1.csv`: selección cuantitativa de fondos alineados con la tesis Asia ex Japan.
- `dataset/factores_unificados.csv`: tabla unificada de factores diarios usada en el pipeline.

### Salidas de la Parte 2

- `dataset/outliers_clustering_part2.csv`: fondos excluidos del análisis principal por comportamiento extremo, conservados para auditoría.
- `dataset/graphext_part2_export.csv`: exportación preparada para visualización externa y análisis de clusters.
- `dataset/fondos_candidatos_part2.csv`: universo final de fondos candidatos del clúster objetivo.

### Salidas de la validación final

- `dataset/aaxj_features_final_test.csv`: features calculadas para AAXJ con la misma lógica de la Parte 1, usadas en la validación de la Parte 2.

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

## Ejecución

Para trabajar con los notebooks:

```powershell
jupyter notebook
```

Después, abre el cuaderno que corresponda dentro de `src/`.

Si quieres regenerar solo la validación final de AAXJ sin abrir Jupyter:

```powershell
python src/aaxj_feature_builder.py
```

## Notas

- Los notebooks usan rutas relativas al proyecto, por lo que conviene ejecutarlos desde la raíz del repositorio.
- `dataset/X_clust_part1.csv` ya está estandarizado y no debe volver a escalarse antes de la Parte 2.
- El script `src/aaxj_feature_builder.py` obtiene datos desde la web oficial de iShares, por lo que necesita conexión a internet en tiempo de ejecución.
