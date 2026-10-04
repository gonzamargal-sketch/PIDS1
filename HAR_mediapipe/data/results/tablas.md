# Tablas de la comparativa de modelos (generado por plot_results.py)

Todos los números salen de `comparison.csv` / `comparison_summary.csv`. Normalización L0_size; media ± std entre 3 semillas. Coste medido en el modelo de split; latencia con batch = 1 en CPU.

## Tabla comparativa principal

| Modelo | Parámetros | Tamaño (KB) | T. entrenamiento (s) | Latencia mediana / p95 (ms) | Throughput (muestras/s) | Acc split (%) | Acc LOPO (%) | Peor persona LOPO (%) | F1 macro LOPO |
|---|---|---|---|---|---|---|---|---|---|
| CNN1 | 21.797 | 287,0 | 18,0 | 4,131 / 5,893 | 44.411 | 98,64 ± 0,34 | 98,15 ± 0,15 | 97,44 | 0,9815 |
| SVM | 4.395 | 40,9 | 2,2 | 0,156 / 0,360 | 124.268 | 98,31 ± 0,00 | 97,65 ± 0,32 | 94,87 | 0,9764 |
| MLP | 4.997 | 87,8 | 14,9 | 2,075 / 2,902 | 87.938 | 98,31 ± 0,34 | 97,88 ± 0,27 | 96,22 | 0,9788 |
| VIT | 18.149 | 352,0 | 51,2 | 27,811 / 38,509 | 3.058 | 98,08 ± 0,71 | 97,24 ± 0,32 | 95,41 | 0,9725 |

## Intervalos de confianza al 95 % (CalculateCI)

| Modelo | Acc split (%) ± IC | Acc LOPO (%) ± IC | F1 macro split |
|---|---|---|---|
| CNN1 | 98,64 ± 1,31 | 98,15 ± 0,84 | 0,9865 |
| SVM | 98,31 ± 1,47 | 97,65 ± 0,94 | 0,9831 |
| MLP | 98,31 ± 1,47 | 97,88 ± 0,90 | 0,9831 |
| VIT | 98,08 ± 1,55 | 97,24 ± 1,02 | 0,9810 |

## Accuracy LOPO por persona (%, media de semillas)

| Persona | CNN1 | SVM | MLP | VIT |
|---|---|---|---|---|
| Gonzalo | 97,72 | 97,98 | 97,85 | 98,25 |
| Iria | 98,25 | 98,11 | 98,25 | 95,55 |
| Marta | 97,44 | 94,87 | 96,22 | 95,95 |
| Miguel | 99,20 | 99,60 | 99,20 | 99,20 |

## Normalización × modelo (accuracy LOPO %, semilla 2022; entre paréntesis, peor persona)

| Normalización | CNN1 | SVM | MLP | VIT |
|---|---|---|---|---|
| None | 97,17 (91,50) | 89,71 (73,68) | 97,07 (91,90) | 88,90 (68,83) |
| L0 | 96,37 (89,07) | 97,07 (95,14) | 98,08 (95,95) | 94,85 (87,04) |
| L0_size | 98,28 (97,57) | 97,78 (95,55) | 97,78 (95,55) | 97,38 (95,55) |

## Matriz de confusión LOPO agregada — CNN1 (nº de muestras; filas = clase real)

| Real \ Predicha | OK | dislike | guerra | like | paz |
|---|---|---|---|---|---|
| OK | 576 | 5 | 8 | 1 | 4 |
| dislike | 2 | 589 | 0 | 0 | 0 |
| guerra | 1 | 0 | 587 | 6 | 6 |
| like | 7 | 0 | 10 | 572 | 5 |
| paz | 0 | 0 | 0 | 0 | 594 |

## Matriz de confusión LOPO agregada — VIT (nº de muestras; filas = clase real)

| Real \ Predicha | OK | dislike | guerra | like | paz |
|---|---|---|---|---|---|
| OK | 572 | 8 | 7 | 4 | 3 |
| dislike | 2 | 589 | 0 | 0 | 0 |
| guerra | 0 | 1 | 588 | 6 | 5 |
| like | 9 | 4 | 14 | 565 | 2 |
| paz | 1 | 0 | 15 | 1 | 577 |
