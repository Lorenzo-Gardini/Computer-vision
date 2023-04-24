import keras
from keras.layers import Dense, Lambda
from keras.models import Sequential
from scikeras.wrappers import KerasClassifier
from utils import *
from detectors import *
from classifiers import SequentialAdaBoostClassifier
from extraction import *
import numpy as np
import tensorflow as tf
import os

tf.keras.utils.disable_interactive_logging()

#unzip("datasets/test-public-faces.zip")
#unzip("datasets/test-public-lists.zip")
#unzip("datasets/test-private-faces.zip")
#unzip("datasets/test-private-lists.zip")
#unzip("datasets/test-private-labels.zip")

train_lists_folder = 'test-public-lists'
train_faces_folder = 'test-public-faces'
train_faces_folder_dst = 'train-faces'
test_lists_folder = 'test-private-lists'
test_labels_folder = 'test-private-labels'
test_faces_folder = "test-private-faces"

#persist_classes_encoding(train_lists_folder, "class_encoding.json")
class_encodings = load_classes_encoding('class_encoding.json')
#move_and_rename_images(train_lists_folder, train_faces_folder, train_faces_folder_dst)

train_resnet_embeddings = create_embeddings_resnet(train_faces_folder_dst, 'train_embeddings_resnet.bin')
train_facenet_embedding = create_embeddings_facenet(train_faces_folder_dst, 'train_embeddings_facenet.bin')

multiclass_train_dataset = \
    create_multiclass_train_dataset(train_lists_folder, class_encodings, train_faces_folder, "train_multiclass.csv")
combine_dataset_embeddings(multiclass_train_dataset, train_resnet_embeddings, "train_resnet_multiclass.bin")
combine_dataset_embeddings(multiclass_train_dataset, train_facenet_embedding, "train_facenet_multiclass.bin")
del multiclass_train_dataset

binary_train_dataset = \
    create_binary_train_dataset('train_multiclass.csv', 'train_binary.csv')
combine_dataset_embeddings(binary_train_dataset, train_resnet_embeddings, "train_resnet_binary.bin")
combine_dataset_embeddings(binary_train_dataset, train_facenet_embedding, "train_facenet_binary.bin")
del binary_train_dataset

del train_resnet_embeddings
del train_facenet_embedding
test_resnet_embeddings = create_embeddings_resnet(test_faces_folder, "test_embeddings_resnet.bin")
test_facenet_embeddings = create_embeddings_facenet(test_faces_folder, "test_embeddings_facenet.bin")

multiclass_test_dataset = \
    create_multiclass_test_dataset(test_lists_folder, test_labels_folder, class_encodings, "test_multiclass.csv")
combine_dataset_embeddings(multiclass_test_dataset, test_resnet_embeddings, "train_resnet_multiclass.bin")
combine_dataset_embeddings(multiclass_test_dataset, test_facenet_embeddings, "train_facenet_multiclass.bin")
del multiclass_test_dataset

binary_test_dataset = \
    create_binary_test_dataset(test_lists_folder, test_labels_folder, "test_binary.csv")
combine_dataset_embeddings(binary_test_dataset, test_resnet_embeddings, "train_resnet_binary.bin")
combine_dataset_embeddings(binary_test_dataset, test_facenet_embeddings, "train_facenet_binary.bin")
del binary_test_dataset

# landscape = load_image("resources/family.jpeg")
# landscape_resized = resize_with_borders(landscape, 512)
# print(landscape_resized.shape)
# show_image(landscape_resized)
#
# portrait = load_image("resources/portrait.jpg")
# portrait_resized = resize_with_borders(portrait, 512)
# print(portrait_resized.shape)
# show_image(portrait_resized)
#
# detector = FaceDetector(landscape)
# landscape_highlighted = detector.highlight_faces()
# show_image(landscape_highlighted)
# for face in detector.crop_faces():
#     show_image(face)
# for pair in pair_images(detector.crop_faces()):
#     show_image(np.hstack([resize_with_borders(pair[0], 100), resize_with_borders(pair[1], 100)]))
#
# model1 = Sequential([
#     keras.Input(1),
#     Lambda(lambda x: x * 2),
#     Dense(50),
#     Dense(1)
# ])
# model2 = Sequential([
#     keras.Input(1),
#     Lambda(lambda x: x / 2),
#     Dense(50),
#     Dense(1)
# ])
# model1.compile(loss="mse")
# model2.compile(loss="mse")
# clf = SequentialAdaBoostClassifier(
#     [
#         KerasClassifier(model=model1, epochs=100, random_state=0xDEADBEEF),
#         KerasClassifier(model=model2, epochs=100, random_state=0xDEADBEEF)
#     ],
#     learning_rate=0.01,
#     random_state=0xDEADBEEF
# )
# clf.fit([[0], [100], [0], [100]], [0, 1, 0, 1])
# print(clf.estimators_[0].model.summary())
# print("\n")
# print(clf.estimators_[1].model.summary())
# print("\n")
# print(clf.predict([[0], [100]]))
