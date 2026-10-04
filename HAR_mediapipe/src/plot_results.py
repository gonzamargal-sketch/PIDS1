# Fase 2 de la comparativa: figuras y tablas a partir de los resultados de compare_models.py.
# Lee SOLO los CSV/JSON/NPY generados en la Fase 1 (no entrena ni evalua nada).
#
# Run from the root of the repository:
#   python HAR_mediapipe/src/plot_results.py
#
# Salidas:
#   HAR_mediapipe/data/results/figures/*.png      las 6 figuras (150 dpi, textos en español)
#   HAR_mediapipe/data/results/tablas.md          tablas en Markdown (para RESULTADOS.md / presentaciones)
#   HAR_mediapipe/data/results/tabla_*.csv        las mismas tablas en CSV (separador ';' y coma decimal, para Excel)
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

RESULTS_DIR = 'HAR_mediapipe/data/results/'
ARTIFACTS_DIR = RESULTS_DIR + 'comparison/'
FIGURES_DIR = RESULTS_DIR + 'figures/'
CLASSES_FILE = 'HAR_mediapipe/data/dataset_pids_gonzalo/train_classes_list.txt'

MODELS = ['CNN1', 'SVM', 'MLP', 'VIT']
NORMS = ['None', 'L0', 'L0_size']
MAIN_NORM = 'L0_size'
CURVES_SEED = 2022
DPI = 150

# Misma color por modelo en todas las figuras (paleta categorica, orden fijo)
MODEL_COLORS = {'CNN1': '#2a78d6', 'SVM': '#eb6834', 'MLP': '#1baf7a', 'VIT': '#eda100'}
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
GRID = '#e4e3df'
SURFACE = '#fcfcfb'
# Rampa secuencial azul (claro -> oscuro) para los heatmaps
BLUES = LinearSegmentedColormap.from_list(
    'blues', ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': TEXT_2, 'xtick.color': TEXT_2, 'ytick.color': TEXT_2,
    'text.color': TEXT, 'axes.titleweight': 'bold', 'axes.titlesize': 12, 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
    'axes.formatter.use_locale': False,
})


def Num(value, decimals=2):
    # Formato español: coma decimal
    return f'{value:.{decimals}f}'.replace('.', ',')


def Thousands(value):
    return f'{int(round(value)):,}'.replace(',', '.')


def PersonName(dataset):
    return dataset.replace('dataset_pids_', '').capitalize()


def ReadCSV(name):
    # keep_default_na=False: si no, pandas lee la normalizacion 'None' como NaN
    return pd.read_csv(RESULTS_DIR + name, keep_default_na=False, na_values=[''])


def Save(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(FIGURES_DIR + name, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'[FIGURE][{FIGURES_DIR + name}]')


def SummaryTable(summary):
    # Una fila por modelo: coste (del split, que es el modelo que se guarda) + precision split y LOPO
    s = summary[(summary.norm == MAIN_NORM) & (summary.person == 'ALL')].set_index(['model', 'protocol'])
    rows = []
    for m in MODELS:
        sp, lo = s.loc[(m, 'split')], s.loc[(m, 'lopo')]
        rows.append({'model': m, 'params': sp.params_mean, 'disk_kb': sp.disk_kb_mean,
                     'train_time_s': sp.train_time_s_mean,
                     'latency_median_ms': sp.latency_median_ms_mean, 'latency_p95_ms': sp.latency_p95_ms_mean,
                     'throughput_sps': sp.throughput_sps_mean,
                     'acc_split': sp.accuracy_mean, 'acc_split_std': sp.accuracy_std, 'ci_split': sp.ci_mean,
                     'acc_lopo': lo.accuracy_mean, 'acc_lopo_std': lo.accuracy_std, 'ci_lopo': lo.ci_mean,
                     'worst_person_lopo': lo.lopo_person_min_mean,
                     'f1_split': sp.f1_macro_mean, 'f1_lopo': lo.f1_macro_mean,
                     'num_seeds': int(lo.num_seeds)})
    return pd.DataFrame(rows).set_index('model')


def PlotTradeoff(table):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sizes = 60 + 2.2 * table.disk_kb  # area proporcional al tamaño en disco
    for m in MODELS:
        r = table.loc[m]
        ax.errorbar(r.latency_median_ms, r.acc_lopo, yerr=r.acc_lopo_std, fmt='none',
                    ecolor=MODEL_COLORS[m], elinewidth=1.5, capsize=4, zorder=2)
        ax.scatter(r.latency_median_ms, r.acc_lopo, s=sizes[m], color=MODEL_COLORS[m],
                   edgecolor=SURFACE, linewidth=2, zorder=3, alpha=0.9)
        # etiqueta a la derecha; la del MLP algo mas baja para no pisar la barra de error de la CNN1
        offset, ha = ((18, -16), 'left') if m == 'MLP' else ((18, -4), 'left')
        ax.annotate(f'{m}\n{Thousands(r.params)} parámetros · {Num(r.disk_kb, 0)} KB',
                    (r.latency_median_ms, r.acc_lopo), xytext=offset, textcoords='offset points',
                    fontsize=9, color=TEXT, ha=ha, va='center')
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: Num(v, 2 if v < 1 else 0)))
    ax.set_xlim(table.latency_median_ms.min() / 2.5, table.latency_median_ms.max() * 6)
    lo = (table.acc_lopo - table.acc_lopo_std).min()
    hi = (table.acc_lopo + table.acc_lopo_std).max()
    ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.set_xlabel('Latencia de inferencia, batch = 1 (ms, mediana, escala log)')
    ax.set_ylabel('Accuracy LOPO (%)')
    ax.set_title('Precisión frente a coste: accuracy LOPO vs latencia')
    ax.grid(True, which='major', color=GRID, linewidth=0.8)
    ax.text(0.01, 0.01, f'Tamaño del punto ∝ tamaño en disco · barras = std entre {int(table.num_seeds.iloc[0])} semillas',
            transform=ax.transAxes, fontsize=8, color=TEXT_2)
    Save(fig, 'tradeoff.png')


def PlotSplitVsLopo(table):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(MODELS))
    w = 0.36
    for i, m in enumerate(MODELS):
        r = table.loc[m]
        for offset, acc, std, hatch, label in [(-w / 2, r.acc_split, r.acc_split_std, None, 'Split'),
                                               (w / 2, r.acc_lopo, r.acc_lopo_std, '////', 'LOPO')]:
            ax.bar(x[i] + offset, acc, w - 0.03, color=MODEL_COLORS[m], hatch=hatch,
                   edgecolor=SURFACE if hatch is None else 'white', linewidth=0)
            ax.errorbar(x[i] + offset, acc, yerr=std, fmt='none', ecolor=TEXT_2, capsize=3, elinewidth=1)
            ax.text(x[i] + offset, acc + std + 0.08, Num(acc), ha='center', va='bottom', fontsize=8, color=TEXT)
    ax.set_xticks(x, MODELS)
    ymin = 95
    ax.set_ylim(ymin, 100)
    ax.set_ylabel(f'Accuracy (%) — eje Y desde {ymin} %')
    ax.set_title('Accuracy en split y en LOPO (media ± std entre semillas)')
    ax.grid(True, axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor='#9a9893', label='Split (test de las 4 personas)'),
                       Patch(facecolor='#9a9893', hatch='////', edgecolor='white', label='LOPO (persona nueva)')],
              frameon=False, loc='upper right', fontsize=9)
    Save(fig, 'accuracy_split_vs_lopo.png')


def Heatmap(ax, data, row_labels, col_labels, fmt, vmin, vmax, sub=None):
    im = ax.imshow(data, cmap=BLUES, vmin=vmin, vmax=vmax, aspect='auto')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            dark = (v - vmin) / (vmax - vmin) > 0.55
            text = fmt(v) + ('' if sub is None else f'\n{sub[i][j]}')
            ax.text(j, i, text, ha='center', va='center', fontsize=9,
                    color='white' if dark else TEXT)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # separacion de 2 px entre celdas
    ax.set_xticks(np.arange(-0.5, data.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0]), minor=True)
    ax.grid(which='minor', color=SURFACE, linewidth=2)
    ax.tick_params(which='minor', length=0)
    return im


def PlotLopoPerPerson(comparison):
    d = comparison[(comparison.norm == MAIN_NORM) & (comparison.protocol == 'lopo') & (comparison.person != 'ALL')]
    pivot = d.pivot_table(index='person', columns='model', values='accuracy', aggfunc='mean')[MODELS]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    im = Heatmap(ax, pivot.values, [PersonName(p) for p in pivot.index], MODELS,
                 lambda v: Num(v), vmin=np.floor(pivot.values.min()) - 1, vmax=100)
    fig.colorbar(im, ax=ax, label='Accuracy LOPO (%)', shrink=0.85)
    ax.set_title('Accuracy LOPO por persona (persona de test × modelo)')
    ax.set_xlabel(f'Media de {d.seed.nunique()} semillas · normalización {MAIN_NORM}', fontsize=8)
    Save(fig, 'lopo_por_persona.png')
    return pivot


def PlotConfusion(model, classes, seeds):
    # Matriz LOPO agregada (suma de las 4 personas y de todas las semillas), normalizada por filas
    cm = sum(np.load(ARTIFACTS_DIR + f'cm_{model}_{MAIN_NORM}_lopo_s{s}.npy') for s in seeds)
    cm_norm = 100 * cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = Heatmap(ax, cm_norm, classes, classes, lambda v: Num(v, 1), vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label='% de la clase real', shrink=0.85)
    ax.set_xlabel('Clase predicha')
    ax.set_ylabel('Clase real')
    ax.set_title(f'Matriz de confusión LOPO — {model}')
    ax.text(0, -0.16, f'Normalizada por filas · LOPO agregado, {len(seeds)} semillas · normalización {MAIN_NORM}',
            transform=ax.transAxes, fontsize=8, color=TEXT_2)
    Save(fig, f'cm_{model}.png')
    return cm, cm_norm


def PlotNormalization(comparison):
    d = comparison[(comparison.protocol == 'lopo') & (comparison.person == 'ALL') & (comparison.seed == CURVES_SEED)]
    acc = d.pivot(index='norm', columns='model', values='accuracy').loc[NORMS, MODELS]
    worst = d.pivot(index='norm', columns='model', values='lopo_person_min').loc[NORMS, MODELS]
    sub = [[f'peor: {Num(v)}' for v in row] for row in worst.values]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = Heatmap(ax, acc.values, NORMS, MODELS, lambda v: Num(v), vmin=np.floor(acc.values.min()) - 1, vmax=100, sub=sub)
    fig.colorbar(im, ax=ax, label='Accuracy LOPO (%)', shrink=0.85)
    ax.set_ylabel('Normalización')
    ax.set_title('Accuracy LOPO según normalización y modelo')
    ax.set_xlabel(f'Semilla {CURVES_SEED} · "peor" = accuracy de la peor persona', fontsize=8)
    Save(fig, 'normalizacion_x_modelo.png')
    return acc, worst


def PlotCurves():
    keras_models = ['CNN1', 'MLP', 'VIT']
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex='col')
    for j, m in enumerate(keras_models):
        with open(ARTIFACTS_DIR + f'history_{m}_{MAIN_NORM}_split_s{CURVES_SEED}.json') as f:
            h = json.load(f)
        epochs = np.arange(1, len(h['loss']) + 1)
        best = int(np.argmin(h['val_loss'])) + 1  # pesos restaurados por el early stopping
        for i, (metric, label, scale) in enumerate([('loss', 'Loss (entropía cruzada)', 1),
                                                    ('accuracy', 'Accuracy (%)', 100)]):
            ax = axes[i, j]
            ax.plot(epochs, scale * np.array(h[metric]), color=MODEL_COLORS[m], linewidth=1.6, label='Entrenamiento')
            ax.plot(epochs, scale * np.array(h['val_' + metric]), color=MODEL_COLORS[m], linewidth=1.6,
                    linestyle='--', label='Validación')
            ax.axvline(best, color=TEXT_2, linewidth=0.8, linestyle=':')
            ax.grid(True, color=GRID, linewidth=0.8)
            if i == 0:
                ax.set_title(f'{m}  (mejor época: {best} de {len(epochs)})')
                ax.set_yscale('log')
            if j == 0:
                ax.set_ylabel(label)
            if i == 1:
                ax.set_xlabel('Época')
                ax.set_ylim(max(0, scale * min(min(h['accuracy']), min(h['val_accuracy'])) - 2), 100.5)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.suptitle(f'Curvas de entrenamiento (split, semilla {CURVES_SEED}, normalización {MAIN_NORM})',
                 fontweight='bold')
    fig.text(0.5, -0.01, 'Línea punteada vertical: época con menor val_loss (pesos que restaura el early stopping). '
             'Loss en escala log.', ha='center', fontsize=8, color=TEXT_2)
    fig.tight_layout()
    Save(fig, 'curvas_entrenamiento.png')


def WriteTables(table, norm_acc, norm_worst, person_pivot, cm_best, cm_worst, best, worst, classes):
    md = ['# Tablas de la comparativa de modelos (generado por plot_results.py)', '',
          'Todos los números salen de `comparison.csv` / `comparison_summary.csv`. '
          f'Normalización {MAIN_NORM}; media ± std entre {int(table.num_seeds.iloc[0])} semillas. '
          'Coste medido en el modelo de split; latencia con batch = 1 en CPU.', '',
          '## Tabla comparativa principal', '',
          '| Modelo | Parámetros | Tamaño (KB) | T. entrenamiento (s) | Latencia mediana / p95 (ms) | '
          'Throughput (muestras/s) | Acc split (%) | Acc LOPO (%) | Peor persona LOPO (%) | F1 macro LOPO |',
          '|---|---|---|---|---|---|---|---|---|---|']
    rows_csv = []
    for m in MODELS:
        r = table.loc[m]
        md.append(f'| {m} | {Thousands(r.params)} | {Num(r.disk_kb, 1)} | {Num(r.train_time_s, 1)} | '
                  f'{Num(r.latency_median_ms, 3)} / {Num(r.latency_p95_ms, 3)} | {Thousands(r.throughput_sps)} | '
                  f'{Num(r.acc_split)} ± {Num(r.acc_split_std)} | {Num(r.acc_lopo)} ± {Num(r.acc_lopo_std)} | '
                  f'{Num(r.worst_person_lopo)} | {Num(r.f1_lopo, 4)} |')
        rows_csv.append({'Modelo': m, 'Parámetros': round(r.params), 'Tamaño (KB)': round(r.disk_kb, 1),
                         'T. entrenamiento (s)': round(r.train_time_s, 1),
                         'Latencia mediana (ms)': round(r.latency_median_ms, 3),
                         'Latencia p95 (ms)': round(r.latency_p95_ms, 3),
                         'Throughput (muestras/s)': round(r.throughput_sps),
                         'Acc split (%)': round(r.acc_split, 2), 'Std split': round(r.acc_split_std, 2),
                         'Acc LOPO (%)': round(r.acc_lopo, 2), 'Std LOPO': round(r.acc_lopo_std, 2),
                         'IC95 LOPO (±)': round(r.ci_lopo, 2),
                         'Peor persona LOPO (%)': round(r.worst_person_lopo, 2),
                         'F1 macro split': round(r.f1_split, 4), 'F1 macro LOPO': round(r.f1_lopo, 4)})
    pd.DataFrame(rows_csv).to_csv(RESULTS_DIR + 'tabla_comparativa.csv', sep=';', decimal=',', index=False)

    md += ['', '## Intervalos de confianza al 95 % (CalculateCI)', '',
           '| Modelo | Acc split (%) ± IC | Acc LOPO (%) ± IC | F1 macro split |', '|---|---|---|---|']
    for m in MODELS:
        r = table.loc[m]
        md.append(f'| {m} | {Num(r.acc_split)} ± {Num(r.ci_split)} | {Num(r.acc_lopo)} ± {Num(r.ci_lopo)} | '
                  f'{Num(r.f1_split, 4)} |')

    md += ['', f'## Accuracy LOPO por persona (%, media de semillas)', '',
           '| Persona | ' + ' | '.join(MODELS) + ' |', '|---|' + '---|' * len(MODELS)]
    for p, row in person_pivot.iterrows():
        md.append(f'| {PersonName(p)} | ' + ' | '.join(Num(row[m]) for m in MODELS) + ' |')

    md += ['', f'## Normalización × modelo (accuracy LOPO %, semilla {CURVES_SEED}; entre paréntesis, peor persona)', '',
           '| Normalización | ' + ' | '.join(MODELS) + ' |', '|---|' + '---|' * len(MODELS)]
    rows_csv = []
    for n in NORMS:
        md.append(f'| {n} | ' + ' | '.join(f'{Num(norm_acc.loc[n, m])} ({Num(norm_worst.loc[n, m])})' for m in MODELS) + ' |')
        rows_csv.append({'Normalización': n, **{f'{m} acc LOPO (%)': round(norm_acc.loc[n, m], 2) for m in MODELS},
                         **{f'{m} peor persona (%)': round(norm_worst.loc[n, m], 2) for m in MODELS}})
    pd.DataFrame(rows_csv).to_csv(RESULTS_DIR + 'tabla_normalizacion.csv', sep=';', decimal=',', index=False)

    for name, (cm, cm_norm) in [(best, cm_best), (worst, cm_worst)]:
        md += ['', f'## Matriz de confusión LOPO agregada — {name} (nº de muestras; filas = clase real)', '',
               '| Real \\ Predicha | ' + ' | '.join(classes) + ' |', '|---|' + '---|' * len(classes)]
        for i, c in enumerate(classes):
            md.append(f'| {c} | ' + ' | '.join(str(int(v)) for v in cm[i]) + ' |')

    with open(RESULTS_DIR + 'tablas.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print(f'[TABLES][{RESULTS_DIR}tablas.md][{RESULTS_DIR}tabla_comparativa.csv][{RESULTS_DIR}tabla_normalizacion.csv]')


def main():
    comparison = ReadCSV('comparison.csv')
    summary = ReadCSV('comparison_summary.csv')
    classes = pd.read_csv(CLASSES_FILE, header=None)[0].tolist()
    seeds = sorted(comparison[(comparison.norm == MAIN_NORM) & (comparison.protocol == 'lopo')].seed.unique())

    table = SummaryTable(summary)
    PlotTradeoff(table)
    PlotSplitVsLopo(table)
    person_pivot = PlotLopoPerPerson(comparison)
    best = table.acc_lopo.idxmax()
    worst = table.acc_lopo.idxmin()
    cm_best = PlotConfusion(best, classes, seeds)
    cm_worst = PlotConfusion(worst, classes, seeds)
    norm_acc, norm_worst = PlotNormalization(comparison)
    PlotCurves()
    WriteTables(table, norm_acc, norm_worst, person_pivot, cm_best, cm_worst, best, worst, classes)


if __name__ == '__main__':
    main()
