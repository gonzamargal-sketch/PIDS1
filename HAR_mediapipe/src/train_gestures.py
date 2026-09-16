# Trains a gesture classifier on the landmarks produced by extract_landmarks.py, joining several recorded datasets.
#
# Run from the root of the repository:
#   python HAR_mediapipe/src/train_gestures.py --norm L0_size --network CNN1
#       -> trains on the train/ folders of every dataset, evaluates on their test/ folders
#          and saves HAR_mediapipe/models/pids_gestures_CNN1_L0_size.keras (+ .meta.json for the demo)
#   python HAR_mediapipe/src/train_gestures.py --norm L0_size --network CNN1 --eval lopo
#       -> leave-one-person-out: trains with 3 people and tests with the 4th (nothing is saved)
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.join(os.getcwd(), "common"))

import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from evaluation import CalculateCI
from landmarksLib import ArrangeInputDataForNetwork, NormalizeLandmarks

DATA_PATH = 'HAR_mediapipe/data/'
MODELS_PATH = 'HAR_mediapipe/models/'
RESULTS_FILE = DATA_PATH + 'results/results.txt'

SEED = 2022
BATCH_SIZE = 32
EPOCHS = 300
PATIENCE = 50
DROPOUT = 0.3
NUM_CNN_FEATURES = 16


def LoadDataset(dataset_name, mode):
    dataset_path = DATA_PATH + dataset_name + '/'
    x = np.load(dataset_path + mode + '_dataset.npy')
    # Labels are stored 1..N following the order of the classes list
    y = pd.read_csv(dataset_path + mode + '_dataset_with_labels.csv')['label'].to_numpy().astype(int) - 1
    classes = pd.read_csv(dataset_path + mode + '_classes_list.txt', header=None)[0].tolist()
    return x, y, classes


def LoadAll(datasets):
    data = {}
    classes = None
    for name in datasets:
        for mode in ['train', 'test']:
            x, y, dataset_classes = LoadDataset(name, mode)
            if classes is None:
                classes = dataset_classes
            elif dataset_classes != classes:
                sys.exit(f'[ERROR] {name}/{mode} has classes {dataset_classes}, expected {classes}')
            data[(name, mode)] = (x, y)
    return data, classes


def DefineNetworkModel(num_classes, network_type, num_landmarks=21):
    model = keras.Sequential(name=network_type)
    model.add(layers.Input(shape=(num_landmarks, 2, 1)))
    if network_type == "CNN1":
        model.add(layers.Conv2D(NUM_CNN_FEATURES, (5, 1), padding='same', activation='relu'))
        model.add(layers.Dropout(DROPOUT))
    elif network_type == "CNN2":
        model.add(layers.Conv2D(NUM_CNN_FEATURES, (3, 3), padding='same', activation='sigmoid'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    else:
        raise ValueError(f"Unknown network_type: {network_type}")
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(DROPOUT))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def Train(x_train, y_train, num_classes, network_type):
    keras.utils.set_random_seed(SEED)
    # Early stopping monitors a validation split taken from the training data, so the test set
    # is never used to pick the weights and the reported test accuracy is not optimistic.
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED)

    model = DefineNetworkModel(num_classes, network_type)
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
                  metrics=['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(
        ArrangeInputDataForNetwork(x_fit), keras.utils.to_categorical(y_fit, num_classes),
        validation_data=(ArrangeInputDataForNetwork(x_val), keras.utils.to_categorical(y_val, num_classes)),
        batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, verbose=0, callbacks=[early_stopping])
    print(f'\t[trained {len(history.history["loss"])} epochs, best val_loss = {min(history.history["val_loss"]):.4f}]')
    return model


def Evaluate(model, x, y, classes):
    y_pred = np.argmax(model.predict(ArrangeInputDataForNetwork(x), batch_size=BATCH_SIZE, verbose=0), axis=1)
    acc = 100 * accuracy_score(y, y_pred)
    return {
        'accuracy': acc,
        'ci': CalculateCI(len(y), acc),
        'f1_macro': f1_score(y, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y, y_pred, labels=range(len(classes))),
        'num_samples': len(y),
    }


def PrintResult(title, result, classes):
    print(f'\n[{title}][{result["num_samples"]} samples]')
    print(f'\t[accuracy = {result["accuracy"]:.2f}% ± {result["ci"]:.2f}][f1 macro = {result["f1_macro"]:.4f}]')
    print(pd.DataFrame(result['confusion_matrix'],
                       index=[f'true {c}' for c in classes], columns=classes).to_string())


def SaveResults(lines):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'a') as f:
        f.write('\n'.join(lines) + '\n\n')
    print(f'\n[RESULTS APPENDED][{RESULTS_FILE}]')


def RunSplit(data, datasets, classes, args):
    x_train = np.concatenate([data[(n, 'train')][0] for n in datasets])
    y_train = np.concatenate([data[(n, 'train')][1] for n in datasets])
    x_train = NormalizeLandmarks(x_train, args.norm)

    start = time.time()
    model = Train(x_train, y_train, len(classes), args.network)
    elapsed = time.time() - start

    lines = [f'Datasets: {", ".join(datasets)}', f'Evaluation: split',
             f'Model Type: {args.network}', f'Normalization Type: {args.norm}']
    x_test_all, y_test_all = [], []
    for name in datasets:
        x_test = NormalizeLandmarks(data[(name, 'test')][0], args.norm)
        y_test = data[(name, 'test')][1]
        x_test_all.append(x_test)
        y_test_all.append(y_test)
        result = Evaluate(model, x_test, y_test, classes)
        print(f'\t[test {name}][accuracy = {result["accuracy"]:.2f}% ± {result["ci"]:.2f}]')
        lines.append(f'Accuracy {name}: {result["accuracy"]:.2f} ± {result["ci"]:.2f}')

    result = Evaluate(model, np.concatenate(x_test_all), np.concatenate(y_test_all), classes)
    PrintResult('TEST ALL', result, classes)
    lines += [f'Accuracy: {result["accuracy"]:.2f} ± {result["ci"]:.2f}',
              f'F-measure (Unweighted): {result["f1_macro"]:.4f}',
              f'Execution Time: {elapsed:.1f} seconds']

    model_filename = os.path.join(MODELS_PATH, f'pids_gestures_{args.network}_{args.norm}.keras')
    model.save(model_filename)
    # The demo reads this file to use the same class order and normalization as in training
    with open(model_filename.replace('.keras', '.meta.json'), 'w') as f:
        json.dump({'classes': classes, 'norm_type': args.norm, 'datasets': datasets}, f, indent=2)
    print(f'[MODEL SAVED][{model_filename}]')
    lines.append(f'Model: {model_filename}')
    SaveResults(lines)


def RunLeaveOnePersonOut(data, datasets, classes, args):
    lines = [f'Datasets: {", ".join(datasets)}', f'Evaluation: leave-one-person-out',
             f'Model Type: {args.network}', f'Normalization Type: {args.norm}']
    total_cm = np.zeros((len(classes), len(classes)), dtype=int)
    accuracies = []
    for held_out in datasets:
        others = [n for n in datasets if n != held_out]
        x_train = np.concatenate([data[(n, m)][0] for n in others for m in ['train', 'test']])
        y_train = np.concatenate([data[(n, m)][1] for n in others for m in ['train', 'test']])
        x_test = np.concatenate([data[(held_out, m)][0] for m in ['train', 'test']])
        y_test = np.concatenate([data[(held_out, m)][1] for m in ['train', 'test']])

        print(f'\n[LOPO][test person = {held_out}]')
        model = Train(NormalizeLandmarks(x_train, args.norm), y_train, len(classes), args.network)
        result = Evaluate(model, NormalizeLandmarks(x_test, args.norm), y_test, classes)
        print(f'\t[accuracy = {result["accuracy"]:.2f}% ± {result["ci"]:.2f}]')
        lines.append(f'Accuracy {held_out}: {result["accuracy"]:.2f} ± {result["ci"]:.2f}')
        accuracies.append(result['accuracy'])
        total_cm += result['confusion_matrix']

    num_samples = int(total_cm.sum())
    overall = 100 * np.trace(total_cm) / num_samples
    tp = np.diag(total_cm)
    f1_macro = np.mean(2 * tp / (total_cm.sum(axis=0) + total_cm.sum(axis=1)))
    PrintResult('LOPO ALL PEOPLE', {'accuracy': overall, 'ci': CalculateCI(num_samples, overall),
                                    'f1_macro': f1_macro, 'confusion_matrix': total_cm,
                                    'num_samples': num_samples}, classes)
    lines.append(f'Accuracy: {overall:.2f} ± {CalculateCI(num_samples, overall):.2f} '
                 f'(per person mean {np.mean(accuracies):.2f}, min {np.min(accuracies):.2f})')
    SaveResults(lines)


def main():
    parser = argparse.ArgumentParser(description='Train a hand gesture classifier from extracted landmarks')
    parser.add_argument('datasets', nargs='*', help='Dataset folders inside HAR_mediapipe/data/ (default: all dataset_pids_*)')
    parser.add_argument('--norm', choices=['None', 'L0', 'L0_size'], default='L0_size')
    parser.add_argument('--network', choices=['CNN1', 'CNN2'], default='CNN1')
    parser.add_argument('--eval', choices=['split', 'lopo'], default='split')
    args = parser.parse_args()

    datasets = args.datasets or sorted(d for d in os.listdir(DATA_PATH) if d.startswith('dataset_pids_'))
    data, classes = LoadAll(datasets)
    print(f'[DATASETS] {datasets}')
    print(f'[CLASSES] {classes}')
    print(f'[SETUP][network = {args.network}][norm = {args.norm}][eval = {args.eval}]')

    if args.eval == 'split':
        RunSplit(data, datasets, classes, args)
    else:
        if len(datasets) < 2:
            sys.exit('[ERROR] leave-one-person-out needs at least 2 datasets')
        RunLeaveOnePersonOut(data, datasets, classes, args)


if __name__ == '__main__':
    main()
