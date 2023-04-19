from mtcnn import MTCNN
import cv2


def _compute_rectangle(bounding_box):
    x1, y1, width, height = bounding_box
    x2 = x1 + width
    y2 = y1 + height
    return x1, y1, x2, y2


def _highlight_face(image, bounding_box, color):
    x1, y1, x2, y2 = _compute_rectangle(bounding_box)
    return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color, 2)


def _crop_face(image, bounding_box):
    x1, y1, x2, y2 = _compute_rectangle(bounding_box)
    return image[y1:y2, x1:x2]


class FaceDetector:

    def __init__(self, image):
        self._image = image
        self._faces = MTCNN().detect_faces(image)

    def highlight_faces(self, color=(255, 204, 102)):
        result_image = self._image
        for face_info in self._faces:
            result_image = _highlight_face(result_image, face_info['box'], color)
        return result_image

    def crop_faces(self):
        return [_crop_face(self._image, face_info['box']) for face_info in self._faces]
