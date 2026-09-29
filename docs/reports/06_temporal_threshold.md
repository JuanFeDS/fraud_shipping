# ⏳ Umbral en el tiempo — Resumen ejecutivo

> Elección del umbral de decisión solo con el pasado, como se haría en producción, y variación de la ganancia entre semanas.
> Detalle completo y gráficos en [`notebooks/06.umbral_temporal.ipynb`](../../notebooks/06.umbral_temporal.ipynb).

## 🎯 Objetivo

En el notebook 04, el umbral se eligió con CV aleatoria (óptimo 0,20, igual al teórico), pero prediciendo las tres últimas semanas un umbral de 0,15 ganaba entre +0,5 y +1,6 puntos en cada una. Esas semanas eran la evaluación, así que no podían usarse para cambiar el umbral. Aquí se elige el umbral **sin CV aleatoria y sin mirar la semana que se evalúa**, para decidir si la política debe cambiar.

## 🧪 Diseño

| | |
|---|---|
| 🤖 Modelo | Pipeline productivo final (19 variables, hiperparámetros del notebook 04); solo se ajusta el umbral |
| 🔮 Predicción forward | Para cada semana (miércoles a martes), el pipeline se entrena con todo lo anterior y predice esa semana |
| 🎚️ Elección | Umbral que maximiza la ganancia en las semanas del 25/03, 01/04 y 08/04 |
| 🧪 Evaluación | Última semana (15/04 al 21/04), que no interviene en la elección; diferencia con bootstrap (1.000 remuestras) |
| ⚖️ Referencia | Predicciones de la CV aleatoria del notebook 04 (semilla 7) **en las mismas filas** |

## 📊 Resultados

| Semana | Tasa de fraude | Forward | CV aleatoria | Aprobar todo |
|---|---|---|---|---|
| 25/03 | 6,1% | 75,3 | 78,7 | 57,4 |
| 01/04 | 5,6% | 73,7 | 73,6 | 54,5 |
| 08/04 | 5,2% | 77,1 | 78,1 | 59,4 |
| 15/04 | 4,3% | 79,5 | 79,9 | 67,1 |
| **Media ± desvío** | | **76,4 ± 2,5** | 77,6 ± 2,7 | 59,6 ± 5,4 |

| Umbral en la última semana | Ganancia |
|---|---|
| Política (0,20) | 79,46% |
| Elegido con las semanas previas (0,12) | 79,53% (+0,09, IC95 de −1,7 a +2,3) |
| Óptimo a posteriori (0,17) | 80,58% |

## 💡 Hallazgos

### 1. 📅 La ganancia varía entre semanas más que entre folds
- Prediciendo cada semana con el pasado, el modelo da **73,7% a 79,5%** (desvío 2,5 puntos), frente al ±1,3 entre folds aleatorios.
- La CV aleatoria varía igual en esas filas (desvío 2,7): la variación es de las semanas (tasa de fraude de 6,1% a 4,3%), no del esquema de validación.
- La expectativa en producción es **~77%, entre 74% y 80% según la semana**, y las alertas de monitoreo deben usar esa variación.

### 2. 🎚️ Elegido con el pasado, el umbral baja, pero en la semana siguiente no se distingue de 0,20
- Con las predicciones forward de tres semanas, el óptimo es **0,12** (+1,5 puntos sobre 0,20 en esas semanas).
- La CV aleatoria en esas mismas filas también elige uno bajo (0,11): son semanas con más fraude. No es solo un efecto de predecir el futuro.
- El óptimo de cada semana es inestable (0,15, 0,14 y 0,12 con forward) y, en la última semana, 0,12 no se distingue de 0,20 (**+0,09 puntos, IC95 de −1,7 a +2,3**): con una semana, la prueba no tiene potencia. El óptimo a posteriori de esa semana es 0,17.

### 3. 📐 Cerca del umbral, el modelo subestima con los dos esquemas
- En la franja 0,10–0,20, forward predice 0,14 y se observa 0,17 (0,22 ponderado por monto). La CV aleatoria, en las mismas filas, también subestima (0,16; 0,20 por monto).

## 🏆 Decisión

| | |
|---|---|
| 🎚️ Umbral | Se mantiene **0,20**: el teórico de la matriz de costos y el óptimo con CV aleatoria. Sale de los costos y no hay que estimarlo; los umbrales elegidos con el pasado no se distinguen de él |
| 📏 Expectativa | ~77% de la ganancia máxima por semana, entre 74% y 80% |

## ⚠️ Supuestos y limitaciones

- Son cuatro semanas de un período atípico: no alcanzan para distinguir un sesgo persistente hacia umbrales bajos de la variación semanal.
- Las semanas con más fraude prefieren umbrales más bajos con cualquier esquema: es un **cambio en la tasa base**. En producción no se persigue el umbral: se corrigen las probabilidades por la tasa vigente, `p′ = p·a / (p·a + (1−p)·b), con a = π′/π y b = (1−π′)/(1−π)`, o se recalibran con etiquetas maduras, y el umbral queda en 0,20.
- La semana del 25/03 se predice con solo 60.000 transacciones de entrenamiento, y pierde 3,4 puntos frente a la CV aleatoria.
