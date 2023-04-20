import json
import os
from itertools import product
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from shutil import copy

from keras_facenet import FaceNet
from keras_vggface.utils import preprocess_input
from keras_vggface import VGGFace
from numpy import expand_dims

from utils import optionally_mkdir, load_image


def get_conf_mapping(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def make_conf_mapping(lists, destination):
    encodings = {Path(file).stem: encoding for encoding, file in enumerate(os.listdir(lists))}
    with open(destination, 'w+') as f:
        json.dump(encodings, f, indent=4)


def move_and_rename_images(lists, source, destination):
    optionally_mkdir(destination)
    for csv_file in os.listdir(lists):
        csv = pd.read_csv(f'{lists}/{csv_file}')
        for _, data in csv.iterrows():
            photos_1 = [data['p1'] + image for image in os.listdir(f"{source}/{data['p1']}")]
            photos_2 = [data['p2'] + image for image in os.listdir(f"{source}/{data['p2']}")]
            for photo in photos_1 + photos_2:
                image_name = photo.replace("/", "_")
                copy(f'{source}/{photo}', f'{destination}/{image_name}')


def create_relational_csv_train(lists, class_encoding, source, destination):
    dataframe = {
        'p1': [],
        'p2': [],
        'relation': []
    }
    for csv_file in os.listdir(lists):
        csv = pd.read_csv(f'{lists}/{csv_file}')
        for _, data in csv.iterrows():
            photos_1 = [(data['p1'] + image).replace("/", "_") for image in os.listdir(f"{source}/{data['p1']}")]
            photos_2 = [(data['p2'] + image).replace("/", "_") for image in os.listdir(f"{source}/{data['p2']}")]
            for photo_1 in photos_1:
                for photo_2 in photos_2:
                    dataframe['p1'].append(photo_1)
                    dataframe['p2'].append(photo_2)
                    dataframe['relation'].append(class_encoding[Path(csv_file).stem])
    pd.DataFrame(dataframe).to_csv(destination, index=False)


def create_binary_csv_train(relational_file, destination):
    csv = pd.read_csv(relational_file)
    dataframe = {
        'p1': [p1 for p1 in csv['p1']],
        'p2': [p2 for p2 in csv['p2']],
        'label': [1] * len(csv)
    }
    couples = {(p1, p2) for p1, p2 in csv.drop(columns='relation').values}
    p1s = np.unique(csv['p1'])
    p2s = np.unique(csv['p2'])

    for p1, p2 in product(p1s, p2s):
        if len(dataframe['p1']) == 2 * len(couples):
            break
        dataframe['p1'].append(p1)
        dataframe['p2'].append(p2)
        dataframe['label'].append(0)

    pd.DataFrame(dataframe).to_csv(destination, index=False)


def create_relational_csv_test(lists, labels, class_encoding, destination):
    dataframe = {
        'p1': [],
        'p2': [],
        'relation': []
    }
    for csv_file in os.listdir(lists):
        csv = pd.read_csv(f'{lists}/{csv_file}')
        csv['labels'] = pd.read_csv(f'{labels}/{csv_file}')
        for _, data in csv.iterrows():
            if data['labels']:
                dataframe['p1'].append(data['p1'])
                dataframe['p2'].append(data['p2'])
                dataframe['relation'].append(class_encoding[Path(csv_file).stem])
    pd.DataFrame(dataframe).to_csv(destination, index=False)


def create_binary_csv_test(lists, labels, destination):
    dataframe = pd.DataFrame()
    for csv_file in os.listdir(lists):
        csv = pd.read_csv(f'{lists}/{csv_file}')
        csv['label'] = pd.read_csv(f'{labels}/{csv_file}')
        dataframe = pd.concat([dataframe, csv])
    pd.DataFrame(dataframe).to_csv(destination, index=False)


def create_embeddings_resnet(source, destination, log_freq=500):
    resnet = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    create_embeddings(source,
                      destination,
                      preprocess_image=lambda x: expand_dims(preprocess_input(x.astype(np.float32), version=2), axis=0),
                      compute_embedding=lambda x: resnet.predict(x)[0],
                      log_freq=log_freq)


def create_embeddings_facenet(source, destination, log_freq=500):
    face_net = FaceNet()
    create_embeddings(source,
                      destination,
                      preprocess_image=lambda x: expand_dims((x - x.mean()) / x.std(), axis=0),
                      compute_embedding=lambda x: face_net.embeddings(x)[0],
                      log_freq=log_freq)


def create_embeddings(source, destination, preprocess_image, compute_embedding, log_freq=500):
    embeddings = {}
    for i, photo in enumerate(os.listdir(source), start=1):
        image = load_image(f'{source}/{photo}')
        preprocessed_image = preprocess_image(image)
        embeddings[photo] = compute_embedding(preprocessed_image)

        if len(embeddings) % log_freq == 0:
            print(len(embeddings))

    with open(destination, 'wb+') as f:
        pickle.dump(embeddings, f)


def load_embedding(source):
    with open(source, 'rb') as f:
        return pickle.load(f)