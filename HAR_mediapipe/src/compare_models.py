# Comparativa de modelos de gestos (CNN1 baseline vs SVM, MLP y VIT) en precision y coste computacional.
# Reutiliza la carga de datos, la CNN y el esquema de entrenamiento de train_gestures.py sin modificarlo.
#
# Run from the root of the repository:
#   python HAR_mediapipe/src/compare_models.py --models CNN1 SVM MLP VIT --norm L0_size --eval both --seeds 2022 2023 2024
#       -> pasada principal: split + leave-one-person-out, 3 semillas
#   python HAR_mediapipe/src/compare_models.py --norm all --eval lopo --seeds 2022
#       -> pasada extra barata: normalizacion x modelo (solo LOPO, 1 semilla)
#
# Salidas (HAR_mediapipe/data/results/):
#   comparison.csv          una fila por modelo, norm, semilla, protocolo y persona (persona = ALL para el agregado)
#   comparison_summary.csv  media ± std sobre las semillas
#   comparison/             matrices de confusion (.npy) e historiales de entrenamiento (.json)
#   results.txt             un bloque por modelo/norm/protocolo, mismo formato que train_gestures.py
# Los modelos SVM/MLP/VIT del split con la semilla 2022 se guardan en HAR_mediapipe/models/ (CNN1 no se guarda,
# para no sobrescribir el modelo que usa la demo).
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.append(os.path.join(os.getcwd(), "common"))

import joblib
import numpy as np
import pandas as pd
import keras
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from evaluation import CalculateCI
from landmarksLib import ArrangeInputDataForNetwork, NormalizeLandmarks
from train_gestures import (BATCH_SIZE, DATA_PATH, EPOCHS, MODELS_PATH, PATIENCE, RESULTS_FILE,
                            DefineNetworkModel, LoadAll, SaveResults)
from gesture_models import DefineMLP, DefineVIT

RESULTS_DIR = DATA_PATH + 'results/'
COMPARISON_CSV = RESULTS_DIR + 'comparison.csv'
SUMMARY_CSV = RESULTS_DIR + 'comparison_summary.csv'
ARTIFACTS_DIR = RESULTS_DIR + 'comparison/'

ALL_MODELS = ['CNN1', 'SVM', 'MLP', 'VIT']
ALL_NORMS = ['None', 'L0', 'L0_size']
KEY_COLUMNS = ['model', 'norm', 'seed', 'protocol', 'person']
SAVE_SEED = 2022

SVM_GRID = {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': ['scale', 0.01, 0.1, 1]}
LATENCY_WARMUP = 50
LATENCY_CALLS = 1000
THROUGHPUT_BATCH = 256
THROUGHPUT_CALLS = 20


def PrepareInput(x, model_type):
    # x: (N, 42) ya normalizado -> forma de entrada de cada modelo
    if model_type == 'CNN1':
        return ArrangeInputDataForNetwork(x).astype(np.float32)          # (N, 21, 2, 1)
    if model_type == 'VIT':
        return ArrangeInputDataForNetwork(x)[..., 0].astype(np.float32)  # (N, 21, 2)
    return x.astype(np.float32)                                          # (N, 42) MLP / SVM


def BuildKerasModel(model_type, num_classes):
    if model_type == 'CNN1':
        return DefineNetworkModel(num_classes, 'CNN1')
    if model_type == 'MLP':
        return DefineMLP(num_classes)
    if model_type == 'VIT':
        return DefineVIT(num_classes)
    raise ValueError(f'Unknown model_type: {model_type}')


def TrainKeras(x_train, y_train, num_classes, model_type, seed, learning_rate):
    # Mismo esquema que Train() de train_gestures.py, pero con la semilla como parametro.
    # Early stopping sobre una validacion sacada del train: el test nunca se usa para elegir pesos.
    keras.utils.set_random_seed(seed)
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.15, stratify=y_train, random_state=seed)

    model = BuildKerasModel(model_type, num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4),
                  metrics=['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(
        PrepareInput(x_fit, model_type), keras.utils.to_categorical(y_fit, num_classes),
        validation_data=(PrepareInput(x_val, model_type), keras.utils.to_categorical(y_val, num_classes)),
        batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, verbose=0, callbacks=[early_stopping])
    epochs = len(history.history['loss'])
    print(f'\t[trained {epochs} epochs, best val_loss = {min(history.history["val_loss"]):.4f}]')
    return model, {k: [float(v) for v in vals] for k, vals in history.history.items()}, {'epochs': epochs}


def TrainSVM(x_train, y_train, seed):
    # Busqueda de hiperparametros con CV interna estratificada SOLO sobre el train
    pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', probability=False))])
    search = GridSearchCV(pipeline, SVM_GRID, scoring='accuracy', n_jobs=-1,
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    search.fit(x_train.astype(np.float32), y_train)
    best = {k.replace('svc__', ''): v for k, v in search.best_params_.items()}
    print(f'\t[best params = {best}, CV accuracy = {100 * search.best_score_:.2f}%]')
    return search.best_estimator_, None, {'best_params': json.dumps(best), 'cv_accuracy': 100 * search.best_score_,
                                          'refit_time_s': search.refit_time_}


def Predict(model, x, model_type):
    if model_type == 'SVM':
        return model.predict(PrepareInput(x, model_type))
    return np.argmax(model.predict(PrepareInput(x, model_type), batch_size=THROUGHPUT_BATCH, verbose=0), axis=1)


def CountParams(model, model_type):
    if model_type == 'SVM':
        svc = model.named_steps['svc']
        # vectores soporte x 42 + coeficientes duales + intercepts
        return int(svc.support_vectors_.size + svc.dual_coef_.size + svc.intercept_.size)
    return int(sum(np.prod(w.shape) for w in model.trainable_weights))


def SaveModel(model, model_type, filename_no_ext):
    filename = filename_no_ext + ('.joblib' if model_type == 'SVM' else '.keras')
    if model_type == 'SVM':
        joblib.dump(model, filename)
    else:
        model.save(filename)
    return filename


def DiskSizeKB(model, model_type):
    with tempfile.TemporaryDirectory() as tmp:
        filename = SaveModel(model, model_type, os.path.join(tmp, 'model'))
        return os.path.getsize(filename) / 1024


def MeasureSpeed(model, x_pool, model_type):
    # Latencia batch=1 (mediana y p95 en ms) y throughput batch=256 (muestras/s), en CPU
    x_pool = PrepareInput(x_pool, model_type)
    x_one = x_pool[:1]
    x_batch = np.resize(x_pool, (THROUGHPUT_BATCH,) + x_pool.shape[1:])
    if model_type == 'SVM':
        call = model.predict
    else:
        x_one = keras.ops.convert_to_tensor(x_one)
        x_batch = keras.ops.convert_to_tensor(x_batch)
        # model(x, training=False) evita el overhead fijo de model.predict
        call = lambda x: model(x, training=False)

    for _ in range(LATENCY_WARMUP):
        call(x_one)
    times = np.empty(LATENCY_CALLS)
    for i in range(LATENCY_CALLS):
        start = time.perf_counter()
        call(x_one)
        times[i] = time.perf_counter() - start

    for _ in range(5):
        call(x_batch)
    start = time.perf_counter()
    for _ in range(THROUGHPUT_CALLS):
        call(x_batch)
    throughput = THROUGHPUT_CALLS * THROUGHPUT_BATCH / (time.perf_counter() - start)
    return {'latency_median_ms': 1000 * np.median(times), 'latency_p95_ms': 1000 * np.percentile(times, 95),
            'throughput_sps': throughput}


def TrainAndMeasure(x_train, y_train, x_speed, num_classes, model_type, seed, args):
    start = time.time()
    if model_type == 'SVM':
        model, history, extra = TrainSVM(x_train, y_train, seed)
    else:
        lr = args.vit_lr if model_type == 'VIT' else 0.001
        model, history, extra = TrainKeras(x_train, y_train, num_classes, model_type, seed, lr)
    cost = {'train_time_s': time.time() - start, 'params': CountParams(model, model_type),
            'disk_kb': DiskSizeKB(model, model_type)}
    cost.update(MeasureSpeed(model, x_speed, model_type))
    cost.update(extra)
    return model, history, cost


def ResultRow(model_type, norm, seed, protocol, person, y, y_pred, extra=None):
    acc = 100 * accuracy_score(y, y_pred)
    row = {'model': model_type, 'norm': norm, 'seed': seed, 'protocol': protocol, 'person': person,
           'num_samples': len(y), 'accuracy': acc, 'ci': CalculateCI(len(y), acc),
           'f1_macro': f1_score(y, y_pred, average='macro')}
    row.update(extra or {})
    return row


def SaveArtifact(name, cm=None, history=None):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    if cm is not None:
        np.save(ARTIFACTS_DIR + f'cm_{name}.npy', cm)
    if history is not None:
        with open(ARTIFACTS_DIR + f'history_{name}.json', 'w') as f:
            json.dump(history, f)


def RunSplit(data, datasets, classes, model_type, norm, seed, args):
    print(f'\n[SPLIT][model = {model_type}][norm = {norm}][seed = {seed}]')
    x_train = NormalizeLandmarks(np.concatenate([data[(n, 'train')][0] for n in datasets]), norm)
    y_train = np.concatenate([data[(n, 'train')][1] for n in datasets])
    x_test = {n: NormalizeLandmarks(data[(n, 'test')][0], norm) for n in datasets}

    model, history, cost = TrainAndMeasure(x_train, y_train, np.concatenate(list(x_test.values())),
                                           len(classes), model_type, seed, args)
    rows, y_all, pred_all = [], [], []
    for name in datasets:
        y = data[(name, 'test')][1]
        y_pred = Predict(model, x_test[name], model_type)
        rows.append(ResultRow(model_type, norm, seed, 'split', name, y, y_pred))
        y_all.append(y)
        pred_all.append(y_pred)
    y_all, pred_all = np.concatenate(y_all), np.concatenate(pred_all)
    rows.append(ResultRow(model_type, norm, seed, 'split', 'ALL', y_all, pred_all, cost))
    print(f'\t[accuracy = {rows[-1]["accuracy"]:.2f}%][f1 = {rows[-1]["f1_macro"]:.4f}]'
          f'[params = {cost["params"]}][latency = {cost["latency_median_ms"]:.3f} ms]')

    tag = f'{model_type}_{norm}_split_s{seed}'
    SaveArtifact(tag, cm=confusion_matrix(y_all, pred_all, labels=range(len(classes))), history=history)

    if model_type != 'CNN1' and seed == SAVE_SEED:
        filename = SaveModel(model, model_type, os.path.join(MODELS_PATH, f'pids_gestures_{model_type}_{norm}'))
        meta = {'classes': classes, 'norm_type': norm, 'datasets': datasets, 'model_type': model_type}
        if model_type == 'SVM':
            meta['best_params'] = json.loads(cost['best_params'])
        with open(os.path.splitext(filename)[0] + '.meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        print(f'[MODEL SAVED][{filename}]')
        rows[-1]['model_file'] = filename
    keras.backend.clear_session()
    return rows


def RunLeaveOnePersonOut(data, datasets, classes, model_type, norm, seed, args):
    print(f'\n[LOPO][model = {model_type}][norm = {norm}][seed = {seed}]')
    rows, costs, y_all, pred_all = [], [], [], []
    for held_out in datasets:
        others = [n for n in datasets if n != held_out]
        x_train = NormalizeLandmarks(np.concatenate([data[(n, m)][0] for n in others for m in ['train', 'test']]), norm)
        y_train = np.concatenate([data[(n, m)][1] for n in others for m in ['train', 'test']])
        x_test = NormalizeLandmarks(np.concatenate([data[(held_out, m)][0] for m in ['train', 'test']]), norm)
        y_test = np.concatenate([data[(held_out, m)][1] for m in ['train', 'test']])

        print(f'\t[test person = {held_out}]')
        model, _, cost = TrainAndMeasure(x_train, y_train, x_test, len(classes), model_type, seed, args)
        y_pred = Predict(model, x_test, model_type)
        rows.append(ResultRow(model_type, norm, seed, 'lopo', held_out, y_test, y_pred, cost))
        print(f'\t[accuracy = {rows[-1]["accuracy"]:.2f}%]')
        costs.append(cost)
        y_all.append(y_test)
        pred_all.append(y_pred)
        keras.backend.clear_session()

    # Agregado: accuracy global (CM sumada), media y peor persona; coste = media de los folds
    # (train_time_s = media por fold; train_time_total_s = suma de los 4 folds)
    person_acc = [r['accuracy'] for r in rows]
    agg_cost = {k: float(np.mean([c[k] for c in costs])) for k in costs[0] if not isinstance(costs[0][k], str)}
    agg_cost['train_time_total_s'] = float(np.sum([c['train_time_s'] for c in costs]))
    agg_cost.update({'lopo_person_mean': float(np.mean(person_acc)), 'lopo_person_min': float(np.min(person_acc))})
    if model_type == 'SVM':
        agg_cost['best_params'] = ' | '.join(c['best_params'] for c in costs)
    y_all, pred_all = np.concatenate(y_all), np.concatenate(pred_all)
    rows.append(ResultRow(model_type, norm, seed, 'lopo', 'ALL', y_all, pred_all, agg_cost))
    print(f'\t[LOPO ALL][accuracy = {rows[-1]["accuracy"]:.2f}%][min person = {np.min(person_acc):.2f}%]')
    SaveArtifact(f'{model_type}_{norm}_lopo_s{seed}', cm=confusion_matrix(y_all, pred_all, labels=range(len(classes))))
    return rows


def ReadComparison():
    # keep_default_na=False: si no, pandas lee la normalizacion 'None' como NaN
    return pd.read_csv(COMPARISON_CSV, keep_default_na=False, na_values=[''])


def UpsertRows(rows):
    # Reemplaza las filas con la misma clave (relanzar no duplica resultados)
    new = pd.DataFrame(rows)
    if os.path.exists(COMPARISON_CSV):
        old = ReadComparison()
        keys = set(map(tuple, new[KEY_COLUMNS].astype(str).values))
        old = old[[tuple(r) not in keys for r in old[KEY_COLUMNS].astype(str).values]]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(KEY_COLUMNS, key=lambda c: c.astype(str)).reset_index(drop=True)
    new.to_csv(COMPARISON_CSV, index=False)


def WriteSummary():
    df = ReadComparison()
    group = ['model', 'norm', 'protocol', 'person']
    numeric = [c for c in df.columns if c not in KEY_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    agg = df.groupby(group)[numeric].agg(['mean', 'std'])
    agg.columns = [f'{c}_{s}' for c, s in agg.columns]
    agg.insert(0, 'num_seeds', df.groupby(group)['seed'].nunique())
    agg.insert(1, 'seeds', df.groupby(group)['seed'].apply(lambda s: ' '.join(map(str, sorted(s.unique())))))
    agg.reset_index().to_csv(SUMMARY_CSV, index=False)
    print(f'[SUMMARY][{SUMMARY_CSV}]')


def ResultsBlock(rows, datasets, model_type, norm, protocol, seeds):
    # Mismo formato que train_gestures.py; con varias semillas se da la media (y la std entre semillas)
    df = pd.DataFrame(rows)
    df = df[(df.model == model_type) & (df.norm == norm) & (df.protocol == protocol)]
    total = df[df.person == 'ALL']
    evaluation = 'split' if protocol == 'split' else 'leave-one-person-out'
    lines = [f'Datasets: {", ".join(datasets)}', f'Evaluation: {evaluation}',
             f'Model Type: {model_type}', f'Normalization Type: {norm}',
             f'Seeds: {", ".join(map(str, seeds))}' + (' (mean over seeds)' if len(seeds) > 1 else '')]
    for name in datasets:
        p = df[df.person == name]
        acc = p.accuracy.mean()
        lines.append(f'Accuracy {name}: {acc:.2f} ± {CalculateCI(int(p.num_samples.iloc[0]), acc):.2f}')
    acc = total.accuracy.mean()
    ci = CalculateCI(int(total.num_samples.iloc[0]), acc)
    std = f', std across seeds {total.accuracy.std():.2f}' if len(seeds) > 1 else ''
    if protocol == 'split':
        lines.append(f'Accuracy: {acc:.2f} ± {ci:.2f}' + (f' ({std[2:]})' if std else ''))
    else:
        lines.append(f'Accuracy: {acc:.2f} ± {ci:.2f} (per person mean {total.lopo_person_mean.mean():.2f}, '
                     f'min {total.lopo_person_min.mean():.2f}{std})')
    lines.append(f'F-measure (Unweighted): {total.f1_macro.mean():.4f}')
    time_col = 'train_time_s' if protocol == 'split' else 'train_time_total_s'
    lines.append(f'Execution Time: {total[time_col].mean():.1f} seconds')
    lines.append(f'Parameters: {int(round(total.params.mean()))}, Disk: {total.disk_kb.mean():.1f} KB, '
                 f'Latency (batch=1): {total.latency_median_ms.mean():.3f} ms median / '
                 f'{total.latency_p95_ms.mean():.3f} ms p95, Throughput (batch={THROUGHPUT_BATCH}): '
                 f'{total.throughput_sps.mean():.0f} samples/s')
    if model_type == 'SVM':
        lines.append(f'Best params: {" | ".join(total.best_params.astype(str))}')
    if 'model_file' in total and total.model_file.notna().any():
        lines.append(f'Model: {total.model_file.dropna().iloc[0]}')
    return lines


def main():
    parser = argparse.ArgumentParser(description='Compare gesture classifiers (accuracy vs computational cost)')
    parser.add_argument('datasets', nargs='*', help='Dataset folders inside HAR_mediapipe/data/ (default: all dataset_pids_*)')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument('--norm', choices=ALL_NORMS + ['all'], default='L0_size')
    parser.add_argument('--eval', choices=['split', 'lopo', 'both'], default='both')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2022, 2023, 2024])
    parser.add_argument('--vit-lr', type=float, default=0.001, help='learning rate del VIT (5e-4 si no converge)')
    args = parser.parse_args()

    datasets = args.datasets or sorted(d for d in os.listdir(DATA_PATH) if d.startswith('dataset_pids_'))
    data, classes = LoadAll(datasets)
    norms = ALL_NORMS if args.norm == 'all' else [args.norm]
    protocols = ['split', 'lopo'] if args.eval == 'both' else [args.eval]
    print(f'[DATASETS] {datasets}')
    print(f'[CLASSES] {classes}')
    print(f'[SETUP][models = {args.models}][norms = {norms}][protocols = {protocols}][seeds = {args.seeds}]')

    all_rows = []
    start = time.time()
    for norm in norms:
        for model_type in args.models:
            for protocol in protocols:
                for seed in args.seeds:
                    if protocol == 'split':
                        rows = RunSplit(data, datasets, classes, model_type, norm, seed, args)
                    else:
                        rows = RunLeaveOnePersonOut(data, datasets, classes, model_type, norm, seed, args)
                    UpsertRows(rows)
                    all_rows += rows
                SaveResults(ResultsBlock(all_rows, datasets, model_type, norm, protocol, args.seeds))
    WriteSummary()
    print(f'\n[DONE][{(time.time() - start) / 60:.1f} min][{COMPARISON_CSV}]')


if __name__ == '__main__':
    main()
