import importlib.util
import cv2
import os
import zipfile
import numpy as np
import pickle
import matplot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

if importlib.util.find_spec("google") is not None and \
        importlib.util.find_spec("google.colab") is not None and \
        importlib.util.find_spec("google.colab.patches") is not None:
    from google.colab.patches import cv2_imshow


    def show_image(image):
        cv2_imshow(image)
else:
    def show_image(image):
        cv2.imshow("", image)
        cv2.waitKey(0)
        cv2.destroyWindow("")


def load_image(path, mode=cv2.IMREAD_COLOR, dtype=np.uint8):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    else:
        return cv2.imread(path, mode).astype(dtype)


def resize_image(image, side):
    old_height, old_width, _ = image.shape
    shape = \
        (side, int(old_height * (side / old_width))) \
            if old_width >= old_height \
            else (int(old_width * (side / old_height)), side)
    return cv2.resize(
        image,
        shape,
        interpolation=(cv2.INTER_AREA if old_height > side or old_width > side else cv2.INTER_CUBIC)
    )


def _compute_rectangle(bounding_box):
    x1, y1, width, height = bounding_box
    x2 = x1 + width
    y2 = y1 + height
    return x1, y1, x2, y2


def highlight_face(image, bounding_box, color):
    x1, y1, x2, y2 = _compute_rectangle(bounding_box)
    return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color, 2)


def crop_face(image, bounding_box):
    x1, y1, x2, y2 = _compute_rectangle(bounding_box)
    return image[y1:y2, x1:x2]


def unzip(path):
    with zipfile.ZipFile(path, "r") as file:
        file.extractall()


def load_binary(path):
    with open(path, "rb") as file:
        result = pickle.load(file)
    return result


def dump_binary(binary, path):
    with open(path, "wb") as file:
        pickle.dump(binary, file)

def plot_images(image_1_path, image_2_path, message):
  image1 = cv2.imread(image_1_path, cv.IMREAD_COLOR)
  image2 = cv2.imread(image_2_path, cv.IMREAD_COLOR)
  image1 = cv2.cvtColor(image1, cv.COLOR_BGR2RGB)
  image2 = cv2.cvtColor(image2, cv.COLOR_BGR2RGB)
  fig, axes = plt.subplots(1, 2)

  axes[0].imshow(image1)
  axes[0].axis('off')
  axes[0].set_title(image_1_path)

  axes[1].imshow(image2)
  axes[1].axis('off')
  axes[1].set_title(image_2_path)

  plt.subplots_adjust(hspace=0.5)
  fig.text(0.5, 0.18, message, ha='center')

  plt.show()

def show_tops(predictions, errors, test_dataframe, give_best=True, is_binary=True, top_n=5, classes=None):
  test_dataframe = test_dataframe.reset_index(drop=True)
  top_values = (errors.argsort()[:top_n] if give_best else errors.argsort()[-top_n:])
  if not give_best:
    top_values = np.flip(top_values)

  for index in top_values.flatten():
    p1 = test_dataframe.at[index, 'p1']
    p2 = test_dataframe.at[index, 'p2']
    res = test_dataframe.at[index, 'label' if is_binary else 'relation']

    if is_binary:
      kinship_percent = f'Predicted {round(predictions[index]*100, 3)}% relationship'
      message = f'{kinship_percent}, {"relationship" if res else "no relationship"}'
    else:
      reversed_classes = {value: key for key, value in classes.items()} 
      prediction_index = predictions[index].argmax()
      prediction_percent = round(predictions[index][prediction_index]*100, 3)
      message = f"""predicted relationship: {reversed_classes[prediction_index]}:{prediction_percent}%, true relationship: {reversed_classes[int(res)]}"""
    plot_images(f'test-private-faces/{p1}', f'test-private-faces/{p2}', message)

def plot_history(history):
    # Extract training metrics from history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Create figure and axis
    _, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    # Plot loss curves
    ax[0].plot(train_loss, label='Train Loss')
    ax[0].plot(val_loss, label='Validation Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # Plot accuracy curves
    ax[1].plot(train_accuracy, label='Train Accuracy')
    ax[1].plot(val_accuracy, label='Validation Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    # Adjust spacing between subplots
    plt.tight_layout()
    # Save the plot if save_file is provided
    plt.show()

def show_model_scores(model, train_pred, train_y, test_pred, test_y, is_binary, classes=None):
    if is_binary:
        train_pred = (train_pred > 0.5).astype(int)
        test_pred = (test_pred > 0.5).astype(int)
        class_labels = ["non-kin", "kin"]
    else:
        train_pred = np.argmax(train_pred, axis=1)
        test_pred = np.argmax(test_pred, axis=1)
        class_labels = sorted(classes, key=classes.get)

    train_score = accuracy_score(train_pred, train_y)
    test_score = accuracy_score(test_pred, test_y)
    model_name = 'binary' if is_binary else 'multiclass'
    print(f"Scores for model {model_name} train score is {train_score}, test score is {test_score}")
    cm = confusion_matrix(test_y, test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f'Confusion matrix for model {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
