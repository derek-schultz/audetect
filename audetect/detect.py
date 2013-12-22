import cv2
import math
import numpy as np

from multiprocessing import Process, Queue

from audetect import conf
from audetect.conf import *
from audetect import utils

class Detector(object):
    def __init__(self, *args, **kwargs):
        self.verbose = kwargs.pop('verbose', False)

    def say(self, message):
        if self.verbose:
            print message

class ActionUnitDetector(Detector):
    def __init__(self, *args, **kwargs):
        super(ActionUnitDetector, self).__init__(self, *args, **kwargs)                
        self.set_sequence(kwargs.pop('sequence', None))
        self.ffdetector = FacialFeatureDetector(verbose=self.verbose)
        self.boosters = []

        # Load all the classifiers once ahead of time because it takes a while
        for number in AU_LABELS.keys():
            if self.verbose:
                self.say("Loading classifier for action unit %d" % (number,))
            booster = cv2.Boost()
            au_data = "%s%d" % ("au", number)
            booster.load(os.path.join(paths.TRAINING_OUTPUT_PATH, au_data))
            self.boosters.append(booster)

    def say(self, message):
        if self.verbose:
            print message

    def set_sequence(self, sequence, type=".png"):
        """
        Accepts a list of numpy arrays (good for webcam data) or a directory
        name containing the images in the sequence (good for testing with 
        Cohn-Kanade)
        """
        self.sequence = []

        if not sequence:
            return

        if isinstance(sequence, basestring):
            files = utils.load_samples(sequence, type)

            images = []
            for image in files:
                image = cv2.imread(image)
                images.append(image)
            self.sequence = images

        elif isinstance(sequence[0], np.array):
            self.sequence = sequence

    def detect(self):
        """ Returns a tuple of active AUs """
        first = self.sequence[0]
        last = self.sequence[-1]

        initial_points = self.ffdetector.locate_features(first)
        final_points = self.ffdetector.locate_features(last)

        aus = self.determine_aus(initial_points, final_points)

        return aus

    def determine_aus(self, initial_points, final_points):
        """
        Runs the AU classifiers on sets of points and returns a tuple of
        the active action units.
        """
        self.say("Determining AUs from acquired facial landmark data")

        booster = cv2.Boost()
        active_aus = []

        for i, booster in enumerate(self.boosters):
            feature_vector = utils.distances(initial_points, final_points)
            feature_vecotr = np.array(feature_vector)
            feature_vector = np.float32(feature_vector)
            guess = booster.predict(feature_vector)

            if guess == True:
                active_aus.append(AU_ZERO_INDEX_MAPPING[i])
            
        return active_aus


class FacialFeatureDetector(Detector):
    def __init__(self, *args, **kwargs):
        super(FacialFeatureDetector, self).__init__(self, *args, **kwargs)
        self.boosters = []

        # Load all the classifiers once ahead of time because it takes a while
        for i in range(len(EXAMINED_POINTS)):
            self.say("Loading classifier for facial feature %d" % (i,))
            booster = cv2.Boost()
            ff_learned_data = "%s%d" % ("feature", i)
            booster.load(os.path.join(paths.TRAINING_OUTPUT_PATH,
                                      ff_learned_data))
            self.boosters.append(booster)

    def locate_features(self, image):
        """
        Given an image of a face, returns a list of the coordinates of the
        facial landmarks.
        IMPORTANT: the image is actually RESIZED and left like that because it
        is late and this is like pre-alpha.
        """
        # Convert to grayscale and extract face data    
        gray, face, scale = utils.preprocess_face_image(image)

        # ROI ratio configuration
        ROIs = {
            'left_eyebrow':
                ((face[1] + face[3] * .15, face[1] + face[3] / 3),
                 (face[0] + face[2] / 8,   face[0] + face[2] / 2)),
            
            'right_eyebrow':
                ((face[1] + face[3] * .15, face[1] + face[3] / 3),
                 (face[0] + face[2] / 2, face[0] + face[2] * 7/8)),

            'left_eye':
                ((face[1] + face[3] / 3, face[1] + face[3] / 2),
                 (face[0] + face[2] / 8, face[0] + face[2] / 2)),

            'left_eye_right_corner':
                ((face[1] + face[3] / 3, face[1] + face[3] / 2),
                 (face[0] + face[2] * 4/12, face[0] + face[2] / 2)),

            'left_eye_left_corner':
                ((face[1] + face[3] / 3, face[1] + face[3] / 2),
                 (face[0] + face[2] / 8, face[0] + face[2] * 4/12)),

            'right_eye_left_corner':
                ((face[1] + face[3] / 3, face[1] + face[3] / 2),
                 (face[0] + face[2] / 2, face[0] + face[2] * 8/12)),

            'right_eye_right_corner':
                ((face[1] + face[3] / 3, face[1] + face[3] / 2),
                 (face[0] + face[2] * 8/12, face[0] + face[2] * 7/8)),
            
            'right_eye':
                ((face[1] + face[3] / 3, face[1] + face[3] / 2),
                 (face[0] + face[2] / 2, face[0] + face[2] * 7/8)),

            'nose':
                ((face[1] + face[3] / 2, face[1] + face[3] * 11/16),
                 (face[0] + face[2] / 4, face[0] + face[2] * 3/4)),
            
            'mouth':
                ((face[1] + face[3] *11/16, face[1] + face[3]),
                 (face[0] + face[2] / 5, face[0] + face[2] * 4/5)),
        }

        if MULTITHREAD:
            queue = Queue()
            classify_threads = []

            # Call the recognize function for the landmark within its ROI
            for landmark, roi in ROI_MAPPING.items():
                roi = ROIs[roi]
                recognizer = Process(target=self.recognize_facial_feature,
                                     args=(gray, landmark, roi, queue))
                classify_threads.append(recognizer)

            for thread in classify_threads:
                thread.start()

            results = []
            for thread in classify_threads:
                results.append(queue.get())
                thread.join()

        else:
            results = []
            for landmark, roi in ROI_MAPPING.items():
                roi = ROIs[roi]
                location = self.recognize_facial_feature(gray, landmark, roi)
                results.append(location)

        return results

    def recognize_facial_feature(self, image, landmark, roi, queue=None):
        """
        Returns the coordinates of the best guess for the location of the
        requested landmark searching within the ROI.
        """
        self.say("Searching for facial feature %d" % (landmark,))

        booster = self.boosters[landmark]

        best = 0
        best_coords = (0,0)
    
        for y in range(int(roi[0][0]), int(roi[0][1])):
            for x in range(int(roi[1][0]), int(roi[1][1])):
                offset_sub = math.floor(PATCH_SIZE / 2.0)
                offset_add = math.ceil(PATCH_SIZE / 2.0)
                sample = image[y - offset_sub : y + offset_add,
                               x - offset_sub : x + offset_add]

                features = utils.apply_filters_to_sample(sample)

                prediction = booster.predict(features)
                confidence = booster.predict(features, returnSum=True)

                if prediction == True:
                    if confidence > best:
                        best = confidence
                        best_coords = (x, y)

        if queue:
            queue.put(best_coords)
        else:
            return best_coords
