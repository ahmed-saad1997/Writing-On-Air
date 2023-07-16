import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import cv2
import math
class Mediapipe_Model():
    def __init__(self):
        self.base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        self.options = vision.HandLandmarkerOptions(base_options=self.base_options,
                                               num_hands=2, running_mode=vision.RunningMode.VIDEO)
        self.model=vision.HandLandmarker.create_from_options(self.options)
        self.timestep=1

    def distance_between_tips(self,finger_landmark1,finger_landmark2):
        distance = math.sqrt((finger_landmark1.x - finger_landmark2.x) ** 2 +
                             (finger_landmark1.y - finger_landmark2.y) ** 2 +
                             (finger_landmark1.z - finger_landmark2.z) ** 2)
        return int(1000*distance)

    def __call__(self,im):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=im)
        results = self.model.detect_for_video(image, timestamp_ms=self.timestep)
        self.timestep+=1
        if len(results.hand_landmarks)==1:
            hand_landmarks = results.hand_landmarks[0]
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            index_fingertip = hand_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            thump_fingertip = hand_landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP]
            middle_fingertip = hand_landmarks[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            dist_indth=self.distance_between_tips(index_fingertip,thump_fingertip)
            dist_indmid = self.distance_between_tips(index_fingertip, middle_fingertip)
            string='break'
            if dist_indth>55 and dist_indmid>55:
                string='write'
            elif dist_indth<=55 and dist_indmid<=55:
                string='predict'
            return index_fingertip,string
        elif len(results.hand_landmarks)==2:
            return None,'clear'
        else:
            return None,'break'