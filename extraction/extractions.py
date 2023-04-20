from extraction.extraction_utility import *
import tensorflow as tf

tf.keras.utils.disable_interactive_logging()

train_lists_folder = '../resources/train/train-public-lists'
train_photo_folder = '../resources/train/train-public-faces-photos'
train_faces_folder = '../resources/train/train-public-faces'
test_lists_folder = '../resources/test/test-private-lists'
test_labels_folder = '../resources/test/test-private-labels'
test_photo_folder = '../resources/test/test-private-faces'


#conf_mapping = get_conf_mapping('../resources/class_encoding.json')

#move_and_rename_images(train_lists_folder, train_faces_folder, train_photo_folder)
#make_conf_mapping(train_lists_folder, 'resources/class_encoding.json')
# create_relational_csv_train(train_lists_folder, conf_mapping, train_faces_folder, "resources/train/train_multiclass.csv")
#create_binary_csv_train('resources/train/train_multiclass.csv', 'resources/train/train_binary.csv')
#create_relational_csv_test(test_lists_folder, test_labels_folder, conf_mapping, "resources/test/test_multiclass.csv")
# create_binary_csv_test(test_lists_folder, test_labels_folder, "resources/test/test_binary.csv")
#create_embeddings_resnet(train_photo_folder, 'resources/train/embeddings', 10)
#create_embeddings_facenet(train_photo_folder, 'resources/train/embeddings/train_embeddings_facenet.bin', 500)
#print(load_embedding('resources/train/embeddings/train_embeddings_facenet.bin'))
