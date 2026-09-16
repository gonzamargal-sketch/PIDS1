# Extracts the hand landmarks of one or more recorded datasets and creates, inside each dataset folder:
#   landmarks/{train,test}/*_poses_landmarks.csv  -> landmarks per class
#   {train,test}_classes_list.txt                 -> class names (label = line number, alphabetical order)
#   {train,test}_dataset_with_labels.csv          -> landmarks + label + frame + path
#   {train,test}_dataset.npy / {train,test}_labels.csv
#
# Run from the root of the repository:
#   python HAR_mediapipe/src/extract_landmarks.py                       (all dataset_pids_* folders)
#   python HAR_mediapipe/src/extract_landmarks.py dataset_pids_gonzalo  (only some datasets)
import argparse
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "common"))

from config import Config, ConfigMediapipeDetector
from landmarksLib import (
    GetLandmarksFromImages,
    extract_classes_list_from_folders,
    sorted_lists_match,
    load_individual_class_features_and_create_labeled_csv_dataset,
    create_numpy_with_feats_and_csv_with_just_labels,
)

DATA_PATH = 'HAR_mediapipe/data/'
DETECTOR_PATH = 'HAR_mediapipe/models/hand_landmarker.task'


def ExtractDataset(detector, dataset_name, config):
    dataset_path = DATA_PATH + dataset_name + '/'
    landmarks_path = dataset_path + 'landmarks/'
    annotations_path = dataset_path + 'annotations/'

    summary = {}
    for mode in ['train', 'test']:
        images_path = dataset_path + mode + '/'
        for images_class in sorted(os.listdir(images_path)):
            image_files = sorted(os.listdir(os.path.join(images_path, images_class)))
            _, num_ok, num_failed = GetLandmarksFromImages(
                detector, image_files, images_path, mode, images_class,
                landmarks_path, annotations_path, config)
            summary[(mode, images_class)] = (num_ok, num_ok + num_failed)

    classes_train = extract_classes_list_from_folders(dataset_path + 'train/', dataset_path)
    classes_test = extract_classes_list_from_folders(dataset_path + 'test/', dataset_path)
    if not sorted_lists_match(classes_train, classes_test):
        sys.exit(f'[ERROR] {dataset_name}: train and test classes differ: {classes_train} vs {classes_test}')

    for mode in ['train', 'test']:
        load_individual_class_features_and_create_labeled_csv_dataset(mode, dataset_path, landmarks_path)
        create_numpy_with_feats_and_csv_with_just_labels(mode, dataset_path)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Extract MediaPipe hand landmarks from recorded datasets')
    parser.add_argument('datasets', nargs='*', help='Dataset folders inside HAR_mediapipe/data/ (default: all dataset_pids_*)')
    parser.add_argument('--save-images', action='store_true', help='Also save the images with the landmarks drawn in annotations/')
    args = parser.parse_args()

    datasets = args.datasets or sorted(d for d in os.listdir(DATA_PATH) if d.startswith('dataset_pids_'))
    config = Config(save_images=args.save_images)
    detector = ConfigMediapipeDetector(DETECTOR_PATH)

    summaries = {name: ExtractDataset(detector, name, config) for name in datasets}

    print('\n[SUMMARY OF SUCCESSFUL DETECTIONS]')
    for name, summary in summaries.items():
        total_ok = sum(ok for ok, _ in summary.values())
        total = sum(n for _, n in summary.values())
        print(f'\n[{name}][{total_ok} out of {total}][{100 * total_ok / total:.2f}%]')
        for (mode, images_class), (ok, n) in summary.items():
            print(f'\t[{mode}][{images_class}][{ok} out of {n}]')


if __name__ == '__main__':
    main()
