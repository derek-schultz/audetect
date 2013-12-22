import cv2
import numpy as np
import random
from multiprocessing import Process

from audetect.conf import *
from audetect import utils

class Trainer(object):
    def __init__(self, *args, **kwargs):
        self.verbose = kwargs.pop('verbose', False)

    def say(self, message):
        if self.verbose:
            print message

class ActionUnitTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(ActionUnitTrainer, self).__init__(self, *args, **kwargs)                
        self.train()

    def train(self):
        for auid in AU_LABELS.keys():
            if MULTITHREAD:
                p = Process(target=self.learn_action_unit, args=(auid,))
                p.start()
            else:
                self.learn_action_unit(auid)

    def learn_action_unit(self, auid):
        booster = cv2.Boost()
        feature_vectors = []

        if auid == 1 and MULTITHREAD:
            self.say("Learning all action units...")
        elif not MULTITHREAD:
            self.say("Learning action unit %d..." % (auid,))

        for session in AU_POSITIVE_SAMPLES[auid]:
            if auid == 1 or not MULTITHREAD:
                self.say("\tLearning session %s" % (session,))
            feature_vector = self.generate_features_from_session(session, auid)
            if feature_vector:
                feature_vectors.append(feature_vector)

        positive_sample_count = len(feature_vectors)

        for session in AU_NEGATIVE_SAMPLES[auid]:
            if auid == 1 or not MULTITHREAD:
                self.say("\tLearning session %s" % (session,))
            feature_vector = self.generate_features_from_session(session, auid)
            if feature_vector:
                feature_vectors.append(feature_vector)
        
        negative_sample_count = len(feature_vectors) - positive_sample_count

        classes = [1 for p in range(positive_sample_count)] + \
                  [0 for n in range(negative_sample_count)]

        var_types = np.array([cv2.CV_VAR_NUMERICAL] * len(feature_vectors[0])\
                             + [cv2.CV_VAR_CATEGORICAL], np.uint8)

        if auid == 1 or not MULTITHREAD:
            self.say("\tBoosting...")

        feature_vectors = np.array(feature_vectors)
        feature_vectors = np.float32(feature_vectors)
        classes = np.array(classes)


        result = booster.train(feature_vectors, cv2.CV_ROW_SAMPLE, classes,
                               varType=var_types, params=BOOST_PARAMS)

        if auid == 1 or not MULTITHREAD:
            self.say("\tBoosted: %s" % (str(result),))

        if not result:
            print len(feature_vectors), feature_vectors
            print len(classes), classes
            exit()

        booster.save(os.path.join(paths.TRAINING_OUTPUT_PATH,
                     "au%d" % (auid,)))
        booster.clear()

    def generate_features_from_session(self, session, auid):
        images_dir = os.path.join(paths.TRAINING_IMAGE_PATH, session)
        landmarks_dir = os.path.join(paths.TRAINING_LANDMARK_PATH, session)
        images = utils.load_samples(images_dir, ".png")
        landmarks = utils.load_samples(landmarks_dir, ".txt")

        if not images or not landmarks:
            print "Invalid session %s on AU %d" % (session, auid)
            exit()

        initial_image = cv2.imread(images[0])
        final_image = cv2.imread(images[-1])

        gray, face, init_scale = utils.preprocess_face_image(initial_image)
        if not face:
            if auid == 1 or not MULTITHREAD:
                self.say("\t%d faces found. Discarding." % (init_scale,))
            return None

        gray, face, final_scale = utils.preprocess_face_image(final_image)
        if not face:
            if auid == 1 or not MULTITHREAD:
                self.say("\t%d faces found. Discarding." % (final_scale,))
            return None

        all_initial_landmarks = utils.load_landmarks(landmarks[0], init_scale)
        all_final_landmarks = utils.load_landmarks(landmarks[-1], final_scale)
        initial_landmarks = []
        final_landmarks = []

        for i in range(len(all_initial_landmarks)):
            if i in EXAMINED_POINTS:
                initial_landmarks.append(all_initial_landmarks[i])
                final_landmarks.append(all_final_landmarks[i])

        distances = utils.distances(initial_landmarks, final_landmarks)
        return distances


class FacialFeatureTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(FacialFeatureTrainer, self).__init__(self, *args, **kwargs)
        
        self.train_size = kwargs.pop('train_size', -1)
        self.images = utils.load_samples(paths.TRAINING_IMAGE_PATH, ".png",
                                         allow_consecutive=False)
        self.landmarks = utils.load_samples(paths.TRAINING_LANDMARK_PATH,
                                            ".txt", allow_consecutive=False)
        
        if self.train_size < 0:
            self.train_size = len(self.images)
        
        self.train()

    def train(self):
        for num, id in enumerate(EXAMINED_POINTS):
            if MULTITHREAD:
                p = Process(target=self.learn_facial_feature, args=(num, id))
                p.start()
            else:
                self.learn_facial_feature(num, id)

    def learn_facial_feature(self, number, point_id):
        booster = cv2.Boost()

        if number == 0 and MULTITHREAD:
            self.say("Learning all facial features...")
        elif not MULTITHREAD:
            self.say("Learning facial feature %d of %d" %
                        (number + 1, len(EXAMINED_POINTS)))

        all_features = []
        all_classes = []

        for i in range(self.train_size):
            if number == 0 or not MULTITHREAD:
                self.say("\tAnalyzing image %s... (%d of %d)" %
                    (os.path.basename(self.images[i]), i+1, self.train_size))

            image = cv2.imread(self.images[i])

            # Convert to grayscale and extract face data    
            gray, face, scale = utils.preprocess_face_image(image)
            if not face:
                if number == 0 or not MULTITHREAD:
                    self.say("\t%d faces found. Discarding." % (scale,))
                continue

            # Load landmarks
            points = utils.load_landmarks(self.landmarks[number], scale=scale)

            # Collect features for the current landmark being trained
            current_landmark = points[point_id]

            # Generate positive sample patches
            positive_samples = []
            offset = PATCH_SIZE / 2
            for x in range(offset, offset - 3, -1):
                for y in range(offset, offset - 3, -1):
                    top = current_landmark[1] - y
                    left = current_landmark[0] - x
                    patch = gray[top:top + PATCH_SIZE, left:left + PATCH_SIZE]
                    positive_samples.append(patch)

            # Randomly choose some negative sample patches
            negative_samples = []
            for i in range(9):
                x = random.randint(PATCH_SIZE, 2 * PATCH_SIZE)
                if random.randint(0, 1):
                    x *= -1
                y = random.randint(PATCH_SIZE, 2 * PATCH_SIZE)
                if random.randint(0, 1):
                    y *= -1
                top = current_landmark[1] + y
                left = current_landmark[0] + x

                patch = gray[top:top + PATCH_SIZE, left:left + PATCH_SIZE]
                negative_samples.append(patch)

            # Build a feature vector for each sample patch
            features = []
            for sample in positive_samples:
                feature_vector = utils.apply_filters_to_sample(sample)
                features.append(feature_vector)
                all_features.append(feature_vector)

            positive_sample_count = len(features)

            for sample in negative_samples:
                feature_vector = utils.apply_filters_to_sample(sample)
                features.append(feature_vector)
                all_features.append(feature_vector)

            negative_sample_count = len(features) - positive_sample_count

            features = np.array(features)
            features = np.float32(features)

            classes = [1 for p in range(positive_sample_count)] + \
                      [0 for n in range(negative_sample_count)]
            classes = np.array(classes)

            all_classes = np.append(all_classes, classes)

        all_features = np.array(all_features)
        all_features = np.float32(all_features)
        all_classes = np.array(all_classes)
        all_classes = np.float32(all_classes)

        var_types = np.array([cv2.CV_VAR_NUMERICAL] * len(features[0])\
                             + [cv2.CV_VAR_CATEGORICAL], np.uint8)

        if number == 0 or not MULTITHREAD:
            self.say("\tBoosting... (this may take a while)")
        booster.train(all_features, cv2.CV_ROW_SAMPLE, all_classes,
                      varType=var_types, params=BOOST_PARAMS)
        booster.save(os.path.join(paths.TRAINING_OUTPUT_PATH,
                                  "feature%d" % (number,)))
        booster.clear()