import cv, cv2
import math
import numpy as np
import os

from audetect.conf import *


def load_samples(directory, type, allow_consecutive=True):
    """
    Returns a list of all files of <type> within and below <directory>.
    allow_consecutive specifies whether multiple samples of the same person
    should be included.
    """
    samples = []
    encountered_faces = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(type):
                face_id = file[0:4]
                if allow_consecutive or face_id not in encountered_faces:
                    encountered_faces.append(face_id)
                    samples.append(os.path.abspath(os.path.join(root, file)))
    return samples


def preprocess_face_image(image):
    """
    Creates an equalized grayscale image, finds faces, and scales so face is
    the correct size. If the number of faces found != 1 then the face returned
    is None and the number of faces found is returned instead of scale for
    error reporting purposes.
    """
    # Equalize histogram
    gray = cv2.cvtColor(image, cv.CV_BGR2GRAY)
    cv2.equalizeHist(gray, gray)

    # Detect faces
    face_classifier = cv2.CascadeClassifier(paths.FACE_HAAR_CASCADE)
    faces = face_classifier.detectMultiScale(gray)

    # If zero or multiple faces are found, discard
    if len(faces) != 1:
        return gray, None, len(faces)

    face = faces[0]

    # Scale the image so the face is specified size
    scale = FACE_SIZE / face[2]
    face = [f * scale for f in face]

    new_image_width = int(len(image[0]) * scale)
    new_image_height = int(len(image) * scale)

    gray = cv2.resize(gray, (new_image_width, new_image_height))
    image = cv2.resize(image, (new_image_width, new_image_height))

    return gray, face, scale


def load_landmarks(file, scale=1):
    points = []
    for data_point in open(file).read().splitlines():
        data_point = data_point[3:]  # Trim padding spaces
        data_point = data_point.split("   ")
        x = int(float(data_point[0]) * scale)
        y = int(float(data_point[1]) * scale)
        points.append((x,y))
    return points


def build_gabor_kernels():
    """
    Returns a list of CV gabor kernels given the parameters for size,
    rotation, and frequency specified in audetect.conf
    """
    kernels = []
    size = (GABOR_FILTER_SIZE, GABOR_FILTER_SIZE)
    for rot in GABOR_ROTATIONS:
        rot = math.pi * rot
        for freq in GABOR_FREQUENCIES:
            kernels.append(cv2.getGaborKernel(size, 5, rot, freq, .5))
    return kernels


def apply_filters_to_sample(sample):
    """
    Given an image patch, applies gabor filters and builds a feature vector.
    """
    # Include the plain grayscale values
    features = []
    features = np.concatenate((features, sample.ravel()))

    # Also include patches with all gabor filters applied
    for kernel in build_gabor_kernels():
        filtered = cv2.filter2D(sample, -1 , kernel).ravel()
        features = np.concatenate((features, filtered))

    features = np.float32(features)
    return features


def distances(initial_points, final_points):
    """
    Calculates the distances between two sets of coordinates. Returned in a
    feature vector friendly format.
    """
    differences = []
    for initial_point, final_point in zip(initial_points, final_points):
        differences.append(final_point[0] - initial_point[0])
        differences.append(final_point[1] - initial_point[1])
    return differences
