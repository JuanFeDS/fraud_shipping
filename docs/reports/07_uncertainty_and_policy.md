# 🧾 Incertidumbre, política de tres zonas y contingencia — Resumen ejecutivo

> Intervalos de confianza de las cifras principales, una política con zona de autenticación (3DS) y un modelo de contingencia sin `score`.
> Detalle completo en [`notebooks/07.incertidumbre_y_politica.ipynb`](../../notebooks/07.incertidumbre_y_politica.ipynb).

## 🎯 Objetivo

Responder tres preguntas que el modelo final deja abiertas: cuánta incertidumbre tienen sus cifras principales, cuánto vale una tercera opción entre aprobar y rechazar, y qué hacer si falta el `score`.

## 🧪 Diseño

| | |
|---|---|
| 🤖 Modelo | Pipeline productivo final; predicciones out-of-fold con la partición del notebook 04 (semilla 7) |
| 📏 Intervalos | Bootstrap de 1.000 remuestras sobre las 150.000 transacciones |
| 🚦 Tres zonas | Aprobar (p < t₁), pedir 3DS (t₁ ≤ p < t₂) o rechazar (p ≥ t₂). En 3DS, una legítima deja su margen si no abandona y un fraude se pierde si no lo frena. Umbrales elegidos con los datos previos al 15/04 y evaluados en la última semana, prediciendo solo con el pasado |
| 🛟 Contingencia | El pipeline con `score` nulo al entrenar y al predecir, frente a aprobar todo y a rechazar por monto |

## 📊 Resultados

| Cifra | Valor | IC95 |
|---|---|---|
| Ganancia del modelo final | 78,9% | 77,6% a 80,1% |
| Ganancia de aprobar todo | 63,4% | 61,3% a 65,5% |
| Mejora frente a aprobar todo | +232 mil | 207 a 258 mil |
| Aporte del tuning | +0,4 puntos | −0,5 a +1,3 |

| Tres zonas en la última semana (5% desafiadas) | Fraude frenado 70% | 85% | 95% |
|---|---|---|---|
| Abandono 5% | +4,5 | +5,9 | +7,0 |
| Abandono 10% | +4,2 | **+5,6** | +6,7 |
| Abandono 20% | +2,6 | +4,9 | +6,0 |

| Contingencia | CV aleatoria | Última semana |
|---|---|---|
| Aprobar todo | 63,4% | 67,1% |
| Rechazar por encima del percentil 80 / 90 / 95 de monto | 28,0% / 39,4% / 48,8% | 29,1% / 40,8% / 49,7% |
| Modelo sin `score` | 77,0% | 77,0% |
| Modelo completo | 78,9% | 79,5% |

## 💡 Hallazgos

### 1. 📏 Las cifras principales son firmes; el aporte del tuning, no
- El 78,9% tiene un IC95 de 77,6% a 80,1%, y los +232 mil frente a aprobar todo, de 207 a 258 mil.
- **El tuning aporta +0,4 puntos con un IC95 que incluye el cero**, y mejora en 3 de los 5 folds. En el modelo final, es compatible con ruido.

### 2. 🚦 Una zona de autenticación es la mejora más grande disponible
- Con el 5% de las transacciones desafiadas, entre ~0,15 y ~0,40, la ganancia sube **entre +2,6 y +7,0 puntos** según los supuestos, y **+5,6 en el escenario central** (10% de abandono, 85% de fraude frenado; IC95 de +4,1 a +7,4).
- Viene de dos lados: las transacciones entre 0,20 y 0,40, que antes se rechazaban, recuperan el margen de las legítimas que se autentican; y las de 0,15 a 0,20, que antes se aprobaban, frenan parte del fraude.
- Cuanto más se desafía, más se gana (+2,8 con 2%, +7,8 con 10%), porque la simulación no cobra la fricción de las legítimas que completan la autenticación ni el costo de 3DS. **El tope de transacciones desafiadas es una decisión de negocio.**

### 3. 🛟 El modelo sin `score` es una contingencia real
- Sin `score`, el modelo da **77,0%**, 1,9 puntos menos que el completo en la CV y 2,5 en la última semana.
- **Rechazar por monto es mucho peor que aprobar todo** (28% a 50%): el 20% más caro concentra la mayor parte de la pérdida por fraude, pero también la mayor parte del margen.

## 🏆 Decisión

| | |
|---|---|
| 🚦 3DS | Candidata para producción: probar en una fracción del tráfico y medir el abandono y el fraude frenado reales antes de fijar la franja |
| 🛟 Respaldo | Modelo sin `score` si falta la variable; aprobar todo (o la regla vigente) si falla el modelo. Nunca un límite de monto |

## ⚠️ Supuestos y limitaciones

- Los parámetros de 3DS (abandono y fraude frenado) son **supuestos**: la mejora depende sobre todo de cuánto fraude frena la autenticación.
- La simulación no incluye el costo por autenticación ni la fricción de las legítimas que la completan; por eso se fija un tope.
- Los intervalos se calculan remuestreando transacciones de la CV aleatoria: no incluyen la variación entre semanas, que es mayor (~2,5 puntos, notebook 06).
