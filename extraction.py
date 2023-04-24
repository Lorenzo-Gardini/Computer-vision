import json
import os
import pickle
from itertools import product
from pathlib import Path
from shutil import copy

import numpy as np
import pandas as pd
from keras_facenet import FaceNet
from keras_vggface import VGGFace
from keras_vggface.utils import preprocess_input

from utils import load_image


def load_classes_encoding(path):
    with open(path, "r") as file:
        return json.load(file)


def persist_classes_encoding(class_files_folder, path):
    encodings = {Path(file).stem: index for index, file in enumerate(os.listdir(class_files_folder))}
    with open(path, 'w') as file:
        json.dump(encodings, file, indent=4)


def move_and_rename_images(class_files_folder, images_source_path, images_destination_path):
    os.makedirs(images_destination_path, exist_ok=True)
    for class_file in os.listdir(class_files_folder):
        csv = pd.read_csv(f'{class_files_folder}/{class_file}')
        for _, data in csv.iterrows():
            images_1 = [data['p1'] + image for image in os.listdir(f"{images_source_path}/{data['p1']}")]
            images_2 = [data['p2'] + image for image in os.listdir(f"{images_source_path}/{data['p2']}")]
            for image in images_1 + images_2:
                image_name = image.replace("/", "_")
                copy(f'{images_source_path}/{image}', f'{images_destination_path}/{image_name}')


def create_multiclass_train_dataset(class_files_folder, class_encoding, images_source_path, dataset_destination_path):
    dataframe = {
        'p1': [],
        'p2': [],
        'relation': []
    }
    for class_file in os.listdir(class_files_folder):
        csv = pd.read_csv(f'{class_files_folder}/{class_file}')
        for _, data in csv.iterrows():
            images_1 = [
                (data['p1'] + image).replace("/", "_") for image in os.listdir(f"{images_source_path}/{data['p1']}")
            ]
            images_2 = [
                (data['p2'] + image).replace("/", "_") for image in os.listdir(f"{images_source_path}/{data['p2']}")
            ]
            for image_1 in images_1:
                for image_2 in images_2:
                    dataframe['p1'].append(image_1)
                    dataframe['p2'].append(image_2)
                    dataframe['relation'].append(class_encoding[Path(class_file).stem])
    pd.DataFrame(dataframe).to_csv(dataset_destination_path, index=False)


def create_binary_train_dataset(multiclass_dataset_path, dataset_destination_path):
    csv = pd.read_csv(multiclass_dataset_path)
    dataframe = {
        'p1': csv['p1'].tolist(),
        'p2': csv['p2'].tolist(),
        'label': [1] * len(csv)
    }
    couples = set((p1, p2) for p1, p2 in csv.drop(columns='relation').values)
    for p1, p2 in product(np.unique(csv['p1']), np.unique(csv['p2'])):
        if len(dataframe) == 2 * len(couples):
            break
        if (p1, p2) not in couples:
            dataframe['p1'].append(p1)
            dataframe['p2'].append(p2)
            dataframe['label'].append(0)
    pd.DataFrame(dataframe).to_csv(dataset_destination_path, index=False)


def create_multiclass_test_dataset(class_files_folder, labels_files_folder, class_encoding, dataset_destination_path):
    dataframe = {
        'p1': [],
        'p2': [],
        'relation': []
    }
    for class_file in os.listdir(class_files_folder):
        csv = pd.read_csv(f'{class_files_folder}/{class_file}')
        csv['labels'] = pd.read_csv(f'{labels_files_folder}/{class_file}')
        for _, data in csv.iterrows():
            if data['labels']:
                dataframe['p1'].append(data['p1'])
                dataframe['p2'].append(data['p2'])
                dataframe['relation'].append(class_encoding[Path(class_file).stem])
    pd.DataFrame(dataframe).to_csv(dataset_destination_path, index=False)


def create_binary_test_dataset(class_files_folder, labels_files_folder, dataset_destination_path):
    dataframe = pd.DataFrame()
    for class_file in os.listdir(class_files_folder):
        csv = pd.read_csv(f'{class_files_folder}/{class_file}')
        csv['label'] = pd.read_csv(f'{labels_files_folder}/{class_file}')
        dataframe = pd.concat([dataframe, csv])
    pd.DataFrame(dataframe).to_csv(dataset_destination_path, index=False)


def create_embeddings_resnet(source_path, destination_path, log_freq=500):
    resnet = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
    _create_embeddings(
        source_path,
        destination_path,
        lambda x: resnet.predict(x)[0],
        log_freq,
        preprocess_image=lambda x: preprocess_input(x.astype(np.float32), version=2)
    )


def create_embeddings_facenet(source_path, destination_path, log_freq=500):
    facenet = FaceNet()
    _create_embeddings(
        source_path,
        destination_path,
        lambda x: facenet.embeddings(x)[0],
        log_freq
    )


def _create_embeddings(source_path, destination_path, compute_embedding, log_freq, preprocess_image=lambda x: x):
    embeddings = {}
    for i, image in enumerate(os.listdir(source_path), start=1):
        image = load_image(f'{source_path}/{image}')
        preprocessed_image = preprocess_image(image)
        embeddings[image] = compute_embedding(preprocessed_image)
        if len(embeddings) % log_freq == 0:
            print(len(embeddings))
    with open(destination_path, 'wb') as file:
        pickle.dump(embeddings, file)
