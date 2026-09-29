# 🧪 Experimentación y modelado — Resumen ejecutivo

> Búsqueda del modelo final: conjuntos de features, comparación de algoritmos, ajuste de hiperparámetros, entrenamiento sensible al monto, simplificación, umbral y validación out-of-time.
> Detalle completo y gráficos en [`notebooks/04.experimentacion.ipynb`](../../notebooks/04.experimentacion.ipynb). Todos los runs quedan registrados en MLflow (`mlflow ui --backend-store-uri sqlite:///mlflow.db`).

## 🎯 Objetivo

Superar al baseline (**77,7% de la ganancia máxima**) con cambios que se sostengan fold a fold, y elegir un modelo final con su umbral de decisión. Se cambia una cosa por vez para poder atribuir cada mejora.

## 🧪 Diseño de los experimentos

| | |
|---|---|
| 📏 Métrica principal | % de la ganancia máxima (+25% del monto por legítima aprobada, −100% por fraude aprobado) |
| 🔁 Validación | Los mismos 5 folds estratificados del baseline; cada experimento se compara **fold a fold** contra una referencia |
| 🎚️ Umbral | El que maximiza la ganancia en la curva out-of-fold |
| 🔤 Tasa de fraude de `j` | Se calcula **dentro de cada fold**, sin usar la etiqueta de la transacción que se predice |
| ✅ Validación del tuning | Folds nuevos (semilla 7) que no intervinieron en la búsqueda de hiperparámetros |

**Orden de los experimentos:**
1. 🎯 Baseline replicado.
2. 🧱 Conjuntos de features con LightGBM por defecto.
3. 🤖 Cinco algoritmos con el mejor conjunto.
4. 🎛️ Búsqueda de hiperparámetros con Optuna (50 pruebas) y validación con folds nuevos.
5. 💰 Entrenamiento con pesos por monto.
6. 🧹 Variables de poco aporte: qué pasa al quitarlas.
7. 🏁 Modelo final validado como pipeline productivo, con su umbral.
8. ⏳ Validación out-of-time: el modelo final prediciendo semanas futuras.
9. 🎚️ Umbral por monto y por país frente al umbral único.

## 📊 Resultados

| Etapa | Mejor configuración | % ganancia máxima | AUC-ROC | AUC-PR | Umbral |
|---|---|---|---|---|---|
| 🎯 Baseline | LightGBM, variables originales | 77,7 ± 0,9 | 0,872 | 0,436 | 0,22 |
| 🧱 Features | LightGBM, conjunto `candidatas` | 79,0 ± 1,0 | 0,889 | 0,467 | 0,16 |
| 🤖 Modelos | LightGBM (empate técnico con CatBoost y Random Forest) | 79,0 ± 1,0 | 0,889 | 0,467 | 0,16 |
| 🎛️ Tuning (folds de la búsqueda) | LightGBM tuneado | 79,7 | — | — | — |
| ✅ Tuning (folds nuevos) | LightGBM tuneado | **78,9 ± 1,4** | 0,890 | 0,474 | 0,15 |
| 💰 Pesos por monto | Sin pesos | 78,9 ± 1,4 | 0,890 | 0,474 | 0,15 |
| 🧹 Simplificación | Sin `perfil_onp` | 79,0 ± 1,3 | 0,890 | 0,474 | 0,17 |
| 🏁 Modelo final | Pipeline productivo sin `perfil_onp` (folds nuevos) | **78,9 ± 1,3** | 0,889 | 0,474 | **0,20** |
| ⏳ Out-of-time | Modelo final, tres últimas semanas | 73,7 / 77,1 / 79,5 | — | — | 0,20 |

## 💡 Hallazgos principales

### 1. 🔤 La `j` cruda era el principal problema del baseline
- Solo con quitarla, el AUC-ROC sube de **0,872 a 0,887**. LightGBM memorizaba sus 8.324 categorías y eso no generaliza.
- **Reemplazarla por su tasa de fraude y su frecuencia** mejora la ganancia en 0,75 puntos y lo hace **en los 5 folds**. Es el cambio individual más importante.
- Mientras la `j` cruda esté presente, el AUC queda en ~0,872 aunque se agreguen su tasa y su frecuencia.

### 2. 🧱 `candidatas` es el mejor conjunto de features
- **79,0% de la ganancia máxima**, +1,2 puntos sobre el baseline **en los 5 folds**, con AUC-PR de 0,467 frente a 0,436.
- Combina `perfil_onp`, `hora`, `g_agrupado` y `j` como tasa y frecuencia. Por separado, `perfil_onp` (+0,24), `hora` (+0,09) y `g_agrupado` (+0,09) aportan poco: LightGBM ya aprendía la interacción entre `o`, `n` y `p` por su cuenta. Más adelante, `perfil_onp` se quita del modelo final (hallazgo 7).
- Agregar las otras 13 features nuevas (`todas_sin_j_cruda`) da el mismo resultado con 33 variables en lugar de 19. Se elige el conjunto más simple.

### 3. 🤖 LightGBM, CatBoost y Random Forest están técnicamente empatados

| Modelo | % ganancia máxima | AUC-ROC | AUC-PR | Folds mejores que LightGBM |
|---|---|---|---|---|
| 🟠 LightGBM | 79,0 | 0,889 | 0,467 | — |
| CatBoost | 78,8 | 0,890 | 0,476 | 2 |
| Random Forest | 78,8 | 0,864 | 0,446 | 2 |
| XGBoost | 77,3 | 0,879 | 0,442 | 1 |
| Regresión logística | 77,1 | 0,847 | 0,382 | 0 |

- La diferencia entre los tres primeros (0,2 puntos) es menor que el desvío entre folds (~1 punto).
- **CatBoost tiene el mejor ranking, pero tarda ~11 minutos por CV** frente a ~6 segundos de LightGBM. Con ese costo no es viable una búsqueda de hiperparámetros.
- **Random Forest gana lo mismo con un AUC bastante menor**: la ganancia depende de acertar en la zona de decisión y en los montos altos, no de ordenar bien todas las transacciones.
- La regresión logística llega casi al baseline de LightGBM: con buenas features, un modelo lineal ya captura la mayor parte de la señal.
- Se elige **LightGBM** por ganancia, velocidad y facilidad para tunear.

### 4. 🎛️ El tuning aporta una mejora real pero modesta
- Sobre los folds de la búsqueda, la mejor prueba llega a 79,7% (+0,7 puntos). Pero **39 de las 50 pruebas superan el 79%**: hay una zona amplia de configuraciones equivalentes, no un óptimo puntual.
- Las mejores configuraciones tienen una **tasa de aprendizaje baja** (~0,02) con más árboles (~480), **hojas grandes** (`num_leaves` ~170) que exigen **muchos casos por hoja** (`min_child_samples` ~260). Son árboles complejos que aprenden despacio, sin hojas tan chicas que permitan memorizar.
- **Con folds nuevos, la mejora baja a +0,4 puntos** (78,9% frente a 78,5%), mejor en 4 de 5 folds. Cerca de la mitad de la mejora de la búsqueda era optimismo por elegir y evaluar con los mismos folds. Que 0,4 puntos cuenten aunque la ganancia varíe ±1 entre folds se debe a que la comparación es **pareada**, fold a fold.
- La partición en sí mueve el resultado ~0,5 puntos: las cifras de ganancia deben leerse con un margen de **±1 punto**.

### 5. 🎚️ Con `perfil_onp`, el umbral óptimo bajaba a 0,15 por la calibración en montos altos
- Las probabilidades están **bien calibradas por conteo**: en el decil más alto, 0,31 predicho frente a 0,31 observado.
- La regla `p < 0,20` no depende del monto: con probabilidades calibradas transacción a transacción, 0,20 sería el óptimo aunque los fraudes sean más caros en promedio.
- Pero en la franja 0,15–0,20, **ponderando por monto**, el fraude observado fue 0,238 frente a 0,173 predicho (0,33 por encima de 300 de monto). El modelo subestimaba el riesgo de las transacciones caras cerca del umbral, y por eso convenía rechazar esa franja.
- La ventaja de 0,15 sobre 0,20 era chica: +0,6 puntos, con IC95 por bootstrap de −0,15 a +1,47.

### 6. 💰 Entrenar con pesos por monto no mejora

| Esquema | % ganancia máxima | AUC-PR | Umbral |
|---|---|---|---|
| Sin pesos | 78,9 | 0,474 | 0,15 |
| Peso = raíz del monto | 78,8 | 0,471 | 0,18 |
| Peso = monto | 78,4 | 0,459 | 0,14 |

- Con la raíz del monto el resultado es el mismo, dentro del ruido. Con el monto directo empeora 0,5 puntos.
- La diferencia se decide en el **quintil de montos más altos**, que concentra más de la mitad de la ganancia. Ahí el modelo con peso por monto rechaza menos y detecta menos fraudes (1.294 frente a 1.366).
- Con montos de 0,02 a 3.696, unas pocas transacciones caras dominan el entrenamiento y el modelo aprende con más varianza. Como `monto` ya es una feature, los pesos no agregan información.

### 7. 🧹 `perfil_onp` no aporta y se quita; `hora` se queda
- Sin `perfil_onp`, el modelo tuneado da **79,0%** (+0,13, mejor en solo 2 de 5 folds, mismo AUC-PR): es ruido. LightGBM ya combina `o`, `n` y `p` por su cuenta. Se quita: la misma ganancia con un modelo más simple.
- Sin `hora`, la ganancia baja 0,26 puntos y el modelo completo es mejor en 4 de 5 folds: su aporte es chico pero consistente, y se queda.
- Se mantienen los hiperparámetros tuneados: el modelo sin perfil ya se evaluó con ellos en los folds nuevos.

### 8. 🏁 Sin `perfil_onp`, el umbral óptimo coincide con el teórico
- Validado como **pipeline productivo** en los folds nuevos (tablas de `j` y de países aprendidas solo con train), el modelo final da **78,9% ± 1,3** con umbral 0,20, y 0,20 es también su óptimo. Con 0,15 da 78,7%.
- La calibración en montos altos mejora: en la franja 0,15–0,20, ponderando por monto, 0,186 observado frente a 0,174 predicho.
- Entre 0,15 y 0,20 la diferencia no es significativa (+0,22, IC95 de −0,33 a +0,71). Se usa **0,20** porque se deduce de la matriz de costos y coincide con el óptimo: no hace falta ajustarlo con los datos.

### 9. ⏳ Prediciendo el futuro, el modelo final pierde poco
- Entrenando solo con el pasado, las tres últimas semanas dan **73,7%, 77,1% y 79,5%**: −0,4 puntos en promedio frente a la CV aleatoria en las mismas filas. La tasa de `j` calculada solo con el pasado no se degrada.
- **Elegir los hiperparámetros con todos los datos no infló el resultado**: con Optuna repetido solo con el pasado y su umbral (0,19), la última semana da 80,0%, frente a 79,5% del modelo final.
- **En el tiempo, un umbral más bajo habría ganado**: con 0,15, entre +0,5 y +1,6 puntos en cada semana. Al predecir el futuro, el modelo parece subestimar el riesgo en la franja de decisión. Se mantiene 0,20 (elegirlo con estas semanas sería usar la evaluación para decidir), pero en producción el umbral debe recalibrarse con etiquetas recientes.

### 10. 🎚️ Un umbral por segmento no mejora de forma concluyente
- Elegidos con el pasado y evaluados en la última semana: por país, −0,01 puntos; por tramo de monto, +0,56 (más estricto entre 100 y 300 de monto). Sobre una sola semana, esa diferencia está dentro del ruido.
- Se mantiene el umbral único. El umbral por tramo de monto es el primer candidato a probar en producción, con más semanas de etiquetas.

## 🏆 Modelo elegido

| | |
|---|---|
| 🌳 Algoritmo | LightGBM tuneado, sin pesos por monto |
| 🧱 Features | Variables originales sin `g` ni `j`, más `g_agrupado`, `hora`, `j_frecuencia` y la tasa de fraude de `j` calculada dentro de cada fold (19 en total) |
| 🎛️ Hiperparámetros | `n_estimators` 481, `learning_rate` 0,019, `num_leaves` 172, `min_child_samples` 262, `subsample` 0,96, `colsample_bytree` 0,96, `reg_alpha` 1,61, `reg_lambda` 3,03 |
| 🎚️ Umbral | 0,20: el teórico de la matriz de costos, que coincide con el óptimo validado |
| 📏 Resultado (pipeline, folds nuevos) | **78,9% ± 1,3 de la ganancia máxima**, AUC-ROC 0,889, AUC-PR 0,474 |
| ⏳ Out-of-time | 73,7%, 77,1% y 79,5% en las tres últimas semanas (−0,4 frente a la CV aleatoria) |
| 📈 Mejora sobre el baseline | ~+1,2 puntos (77,7% → 78,9%) y sobre aprobar todo, ~+15,5 puntos (63,4% → 78,9%) |

## 🛠️ Próximos pasos

- 🔬 Explicar el modelo elegido: qué variables usa, con qué forma y cuánto vale cada una en ganancia (notebook 05).
- 🎚️ En producción, recalibrar el umbral con etiquetas recientes y probar un umbral por tramo de monto con más semanas.

## ⚠️ Supuestos y limitaciones

- La métrica es **ruidosa**: cambios triviales mueven la ganancia ~0,3 puntos porque cambian el umbral elegido. Solo se consideran mejoras las que se repiten en la mayoría de los folds.
- La selección usó **validación aleatoria**. La validación out-of-time del modelo final la respalda (−0,4 puntos en promedio), pero son solo tres semanas de un período atípico, y en el tiempo un umbral más bajo habría ganado.
- `j_frecuencia` y los países frecuentes se calculan con todo el dataset antes de separar los folds en las comparaciones del notebook (no usan la etiqueta). El pipeline productivo lo hace solo con train y reproduce el resultado.
- El resultado depende de que `score` esté disponible al momento de decidir, igual que en el baseline.
- Los algoritmos distintos de LightGBM se compararon con sus **parámetros por defecto**. XGBoost, en particular, podría mejorar con ajuste.
