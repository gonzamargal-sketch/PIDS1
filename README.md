# Reconocimiento de gestos de mano con MediaPipe

PIDS · Gonzalo, Iria, Marta y Miguel

El sistema reconoce 5 gestos estáticos de mano en tiempo real: `OK`, `dislike`, `guerra`, `like` y `paz`. Primero
MediaPipe localiza los 21 puntos clave (landmarks) de la mano. Después una red neuronal pequeña (CNN1) clasifica
esas 42 coordenadas (x, y). Así el modelo es ligero y apenas le afectan el fondo, la iluminación o la ropa.

**Vídeo de la demo:** <https://drive.google.com/file/d/1msIe-u083t4bFogXtcNTiSTVffvQh0nV/view?usp=drivesdk>

**Resultados:** el modelo de la demo (CNN1 con normalización `L0_size`) acierta un 98,98 % en test y un 98,28 % con
una persona que no ha visto en el entrenamiento (leave-one-person-out). La comparativa con SVM, MLP y un Vision
Transformer está en [`HAR_mediapipe/RESULTADOS.md`](HAR_mediapipe/RESULTADOS.md).

## Pipeline

```
Cámara → MediaPipe (21 landmarks) → Normalización L0_size → CNN1 → gesto + confianza
```

- **`L0_size`:** traslada la muñeca (landmark 0) al origen y escala la mano para que la palma (landmark 0 → 9)
  mida 1. Así el resultado no depende de dónde esté la mano en la imagen, de su distancia a la cámara ni de su
  tamaño.
- **Mismo preprocesado en los dos lados:** el entrenamiento y la demo usan la misma función, `NormalizeLandmarks`.
- **`.meta.json`:** cada modelo guarda en él el orden de las clases y la normalización con que se entrenó, y la demo
  los lee de ahí.

## Estructura

```
common/                     utilidades compartidas
  cameras.py                CVCamera (OpenCV/V4L2) y PICamera (Raspberry Pi)
  gui.py                    mensajes en pantalla y lectura de teclado
  evaluation.py             intervalo de confianza al 95 %
HAR_mediapipe/
  src/
    config.py               configuración del detector de MediaPipe y de la grabación
    landmarksLib.py         extracción y normalización de landmarks
    record_dataset.py       1. grabar un dataset con la cámara
    extract_landmarks.py    2. imágenes → landmarks
    train_gestures.py       3. entrenar y evaluar la CNN
    demo-custom-dataset.py  4. demo en directo con la cámara
    compare_models.py       comparativa CNN1 / SVM / MLP / VIT
    gesture_models.py       arquitecturas MLP y VIT
    plot_results.py         figuras y tablas de la comparativa
  models/
    hand_landmarker.task                  detector de mano de MediaPipe
    pids_gestures_CNN1_L0_size.keras      modelo de la demo (+ .meta.json)
    pids_gestures_*                       resto de modelos entrenados
  data/results/             resultados, tablas y figuras
  logs.txt                  imágenes en las que MediaPipe no detectó la mano
  RESULTADOS.md             comparativa de modelos
requirements.txt
```

Los datasets grabados (`HAR_mediapipe/data/dataset_pids_<persona>/`) no se incluyen. Para volver a entrenar hay que
grabarlos con el paso 1.

## Entorno

Los scripts funcionan en **Linux** (Ubuntu en WSL2 o Raspberry Pi OS). En Windows nativo no funcionan, porque
`common/gui.py` usa `termios` y `CVCamera` usa el backend V4L2.

```console
python3 -m venv ~/venvs/ie-workspace
source ~/venvs/ie-workspace/bin/activate
pip install -r requirements.txt
```

**Todos los comandos se ejecutan desde la raíz del repositorio,** porque las rutas son relativas a ella.

En WSL2, la webcam hay que pasarla desde PowerShell como administrador con `usbipd`:

```console
usbipd list
usbipd bind --busid X-Y
usbipd attach --wsl --busid X-Y
```

Después, comprueba en WSL que existe `/dev/video0`.

## Uso

**Demo en directo.** Solo necesita los modelos ya entrenados que hay en `HAR_mediapipe/models/`:

```console
python3 HAR_mediapipe/src/demo-custom-dataset.py
```

Se abre una ventana con el vídeo, los landmarks y el gesto reconocido con su confianza. Pulsa `q` para salir. Al
principio del script se pueden cambiar `MODEL_PATH`, la cámara (`index_cam`), la resolución y `ON_RASPBERRY_PI`.

**Reentrenar desde cero:**

```console
# 1. Grabar: edita antes el bloque Config(...) del script (clases y dataset_dir = HAR_mediapipe/data/dataset_pids_<nombre>/)
python3 HAR_mediapipe/src/record_dataset.py

# 2. Extraer landmarks de todos los dataset_pids_*
python3 HAR_mediapipe/src/extract_landmarks.py

# 3. Entrenar el modelo de la demo (genera .keras + .meta.json)
python3 HAR_mediapipe/src/train_gestures.py --network CNN1 --norm L0_size

#    Evaluación leave-one-person-out (no guarda modelo)
python3 HAR_mediapipe/src/train_gestures.py --network CNN1 --norm L0_size --eval lopo
```

**Comparativa de modelos:**

```console
python3 HAR_mediapipe/src/compare_models.py --models CNN1 SVM MLP VIT --norm L0_size --eval both --seeds 2022 2023 2024
python3 HAR_mediapipe/src/plot_results.py
```
