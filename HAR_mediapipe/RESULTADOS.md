# Comparativa de modelos para el reconocimiento de gestos estáticos de mano

Comparamos la CNN de la demo (**CNN1**) con tres alternativas: **SVM**, **MLP** y un **Vision Transformer (VIT)**
adaptado a landmarks. Medimos precisión y coste computacional.

## Resumen

- **En precisión, los cuatro modelos están prácticamente empatados.** En leave-one-person-out (LOPO) sacan entre
  97,24 % y 98,15 %, y las diferencias son del orden del intervalo de confianza.
- **La ventaja de la CNN1 está en la fiabilidad, más que en la media.** Tiene la mejor peor persona (97,44 %) y la
  menor variación entre semillas.
- **El coste diferencia mucho más que la precisión.** La latencia va de 0,156 ms (SVM) a 27,811 ms (VIT).
- **`L0_size` es la normalización más robusta.** Con ella, la peor persona supera el 95,5 % en todos los modelos.
- **Decisión: mantenemos la CNN1 en la demo.** Sus 4 ms por frame caben de sobra en los 67 ms que da la cámara a
  15 FPS. El SVM sería la alternativa en un hardware más limitado, y el VIT queda descartado.

| Modelo | Acc LOPO (%) | Peor persona (%) | Latencia mediana (ms) | Parámetros |
|---|---|---|---|---|
| CNN1 | 98,15 ± 0,15 | 97,44 | 4,131 | 21.797 |
| SVM | 97,65 ± 0,32 | 94,87 | 0,156 | 4.395 |
| MLP | 97,88 ± 0,27 | 96,22 | 2,075 | 4.997 |
| VIT | 97,24 ± 0,32 | 95,41 | 27,811 | 18.149 |

## 1. Datos y protocolo

- **Dataset:** 4 personas (Gonzalo, Iria, Marta y Miguel) y 5 gestos (`OK`, `dislike`, `guerra`, `like` y `paz`).
  Son 991 muestras balanceadas: 696 de train y 295 de test, unas 50 por clase y persona.
- **Entrada:** los 21 landmarks de MediaPipe × (x, y), es decir, 42 valores por muestra.
- **Evaluación:**
  - **Split:** se entrena con el train de las 4 personas y se evalúa con su test. Mide cómo reconoce el modelo a
    gente conocida.
  - **LOPO:** se entrena con 3 personas y se evalúa con la 4ª, rotando entre las cuatro. Es la métrica que importa
    para la demo, porque mide cómo funciona con alguien que no ha grabado datos.
- **Semillas:** 2022, 2023 y 2024. Se da la media ± la desviación típica.
- **El test no se usa para elegir nada.** Los modelos Keras hacen early stopping (patience 50) sobre un 15 % de
  validación sacado del train. El SVM elige C y gamma con `GridSearchCV` (5 folds) solo sobre el train.
- **Coste:** se mide en CPU sobre el modelo de split. La latencia es la mediana de 1000 llamadas con batch = 1, tras
  50 de calentamiento. El tiempo de entrenamiento del SVM incluye la búsqueda de hiperparámetros.

## 2. Modelos

- **CNN1 (baseline):** Conv2D de 16 filtros (5×1) sobre la secuencia de landmarks → Dense(32) → Dense(5). Cada
  filtro ve 5 puntos consecutivos, por ejemplo las falanges de un mismo dedo.
- **SVM:** `StandardScaler` + `SVC` con kernel RBF sobre el vector de 42 valores.
- **MLP:** Dense(64) → Dense(32) → Dense(5), con dropout 0,3.
- **VIT ("landmark = patch"):** cada landmark se proyecta a un token de dimensión 32. Se añaden un token CLS y un
  embedding posicional, y siguen 2 bloques Transformer con 4 cabezas de atención. Así cualquier punto de la mano
  puede atender a cualquier otro, no solo a sus vecinos.

Las arquitecturas están en `src/train_gestures.py` (CNN1) y `src/gesture_models.py` (MLP y VIT).

## 3. Normalización

| Modo | Qué hace | Accuracy LOPO de CNN1 / SVM / MLP / VIT (%) |
|---|---|---|
| `None` | Coordenadas tal cual, en [0, 1] | 97,17 / 89,71 / 97,07 / 88,90 |
| `L0` | Resta la muñeca: invariante a la posición | 96,37 / 97,07 / 98,08 / 94,85 |
| `L0_size` | L0 + escala por la longitud de la palma: invariante también a la distancia y al tamaño de la mano | 98,28 / 97,78 / 97,78 / 97,38 |

![Normalización × modelo](data/results/figures/normalizacion_x_modelo.png)

- **Lo decisivo es la peor persona.** Sin normalizar, cae al 73,68 % en el SVM y al 68,83 % en el VIT. Con
  `L0_size`, supera el 95,5 % en los cuatro modelos.
- **La tabla usa una sola semilla (2022),** así que diferencias de menos de 1 punto no son concluyentes.
- **Discrepancia con `data/results/results.txt`.** Para CNN1 con `None`, `results.txt` da 95,96 % en LOPO (peor
  persona: Marta, 86,23 %), y esta tabla da 97,17 % (peor persona: 91,50 %). Son dos ejecuciones distintas: la de
  `train_gestures.py` y la de `compare_models.py`. El entrenamiento no es del todo determinista y, sin normalizar, el
  resultado varía más. Las filas `L0` y `L0_size` coinciden en los dos ficheros, y la conclusión no cambia.

## 4. Resultados con `L0_size`

Media ± std de 3 semillas. El IC al 95 % se calcula con `CalculateCI` sobre 295 muestras en split y 991 en LOPO.

| Modelo | Acc split (%) | Acc LOPO (%) | IC LOPO | F1 macro LOPO | Tamaño (KB) | T. entrenamiento (s) | Latencia p95 (ms) | Throughput (muestras/s) |
|---|---|---|---|---|---|---|---|---|
| CNN1 | 98,64 ± 0,34 | 98,15 ± 0,15 | ± 0,84 | 0,9815 | 287,0 | 18,0 | 5,893 | 44.411 |
| SVM | 98,31 ± 0,00 | 97,65 ± 0,32 | ± 0,94 | 0,9764 | 40,9 | 2,2 | 0,360 | 124.268 |
| MLP | 98,31 ± 0,34 | 97,88 ± 0,27 | ± 0,90 | 0,9788 | 87,8 | 14,9 | 2,902 | 87.938 |
| VIT | 98,08 ± 0,71 | 97,24 ± 0,32 | ± 1,02 | 0,9725 | 352,0 | 51,2 | 38,509 | 3.058 |

Accuracy LOPO por persona (%, media de 3 semillas):

| Persona | CNN1 | SVM | MLP | VIT |
|---|---|---|---|---|
| Gonzalo | 97,72 | 97,98 | 97,85 | 98,25 |
| Iria | 98,25 | 98,11 | 98,25 | 95,55 |
| Marta | 97,44 | 94,87 | 96,22 | 95,95 |
| Miguel | 99,20 | 99,60 | 99,20 | 99,20 |

### Figuras

| | |
|---|---|
| ![Tradeoff](data/results/figures/tradeoff.png) **Precisión frente a coste:** la latencia abarca dos órdenes de magnitud y la precisión menos de 1 punto. | ![Split vs LOPO](data/results/figures/accuracy_split_vs_lopo.png) **Split frente a LOPO:** todos pierden entre 0,4 y 0,9 puntos con una persona nueva. |
| ![LOPO por persona](data/results/figures/lopo_por_persona.png) **LOPO por persona:** Marta es la más difícil; solo la CNN1 supera el 97,4 % con las cuatro personas. | ![Curvas](data/results/figures/curvas_entrenamiento.png) **Curvas de entrenamiento:** hay un sobreajuste leve, que el early stopping corta. El VIT es el que más sobreajusta. |
| ![CM CNN1](data/results/figures/cm_CNN1.png) **Matriz de confusión LOPO de la CNN1** (el mejor modelo). | ![CM VIT](data/results/figures/cm_VIT.png) **Matriz de confusión LOPO del VIT** (el peor modelo). |

## 5. Discusión

**¿Hay diferencias reales de precisión?** Prácticamente no. La mayor diferencia en LOPO es la de CNN1 frente a VIT:
0,91 puntos, del orden del IC (±0,84 a ±1,02), y los intervalos se solapan. Además, el IC asume muestras
independientes, y las de una misma persona están correlacionadas, así que la incertidumbre real es mayor. Lo único
consistente es que la CNN1 tiene la mejor peor persona y la menor variación entre semillas.

**¿Qué modelo elegimos?** **La CNN1.**

- **Coste asumible:** a 15 FPS hay unos 67 ms por frame, y la CNN1 tarda 4,1 ms de mediana (5,9 ms en p95).
- **Es la más fiable con una persona nueva.**
- **SVM:** es el más eficiente (26 veces más rápido, 40,9 KB, entrena en 2,2 s), pero tiene la peor persona más
  baja (94,87 %). Sería la opción en un hardware limitado. El MLP queda como término medio si se quiere seguir en
  Keras.
- **VIT:** no compensa. Es 7 veces más lento que la CNN1, tarda casi 3 veces más en entrenar y es el menos preciso.
  Con 21 tokens y menos de 1000 muestras, la atención no aprende nada que no capten ya una convolución o un MLP.

**¿Qué clases se confunden?**

- **`like` es la más difícil.** La CNN1 la confunde 10 veces con `guerra` y 7 con `OK`, porque el pulgar y los dedos
  extendidos ocupan zonas parecidas respecto a la muñeca.
- **`like` y `dislike` casi no se confunden,** aunque son la misma mano girada 180°. `L0_size` traslada y escala, pero
  no rota, así que las dos tienen coordenadas y opuestas. Una normalización de rotación las haría indistinguibles.
- **El VIT confunde `paz` con `guerra` 15 veces** (la CNN1, ninguna). Son gestos que solo se diferencian en qué
  dedos están extendidos.

**¿Por qué no hicimos data augmentation?** La CNN1 ya comete unos 18 errores en 991 muestras, así que el margen de
mejora es menor que el IC. Además, los augmentations típicos para landmarks (traslación y escalado) son justo lo que
`L0_size` ya elimina. La variación que queda viene de cada persona, y eso se arregla mejor grabando a más gente.

## 6. Limitaciones

- **Dataset pequeño:** 4 personas y 991 muestras. Con solo 4 folds, LOPO tiene mucha varianza y no garantiza el
  resultado con una 5ª persona muy distinta.
- **Solo gestos estáticos y en 2D:** un frame por muestra, sin la profundidad z de MediaPipe.
- **Coordenadas no isotrópicas:** MediaPipe normaliza x por el ancho e y por el alto de la imagen, así que la
  longitud de palma de `L0_size` depende algo de la orientación de la mano. Se corregiría multiplicando x por la
  relación de aspecto, pero no se ha hecho para no cambiar la demo.
- **Latencias relativas:** se midieron en un portátil (CPU, WSL2) y en modo eager. Con `tf.function` o TFLite, los
  modelos Keras bajarían bastante, sobre todo el VIT.

## Reproducir

Desde la raíz del repositorio:

```console
python HAR_mediapipe/src/compare_models.py --models CNN1 SVM MLP VIT --norm L0_size --eval both --seeds 2022 2023 2024
python HAR_mediapipe/src/compare_models.py --norm all --eval lopo --seeds 2022
python HAR_mediapipe/src/plot_results.py
```

Los datos brutos están en `data/results/comparison.csv` y `comparison_summary.csv`. Las tablas completas están en
[`data/results/tablas.md`](data/results/tablas.md) y, en CSV para Excel, en `data/results/tabla_*.csv`.
