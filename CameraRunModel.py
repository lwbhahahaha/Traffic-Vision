import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageGrab
import warnings
import time
import tensorflow as tf # Added as colab instance often crash
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import os
import cv2
import math
import cv2
from vidgear.gears import CamGear
import pafy

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

PATH_TO_LABELS = "C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/models/research/object_detection/training/ssd_resnet50_bdd.pbtxt"
PATH_TO_SAVED_MODEL = os.path.join("C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/models/research/object_detection/ssd_resnet50_bdd_with_Stop_Sign/saved_model")
print('Loading model...', end='')
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
print('Done!')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
warnings.filterwarnings('ignore')
def processFrame(image):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    for i in range(len(detections['detection_classes'])):
        if (detections['detection_classes'][i] == 1):
            detections['detection_scores'][i] *= 2.19  # bicycle 3.1 3.05
        elif (detections['detection_classes'][i] == 2):
            detections['detection_scores'][i] *= 2.7  # bus 2.75 2.7 2.6 2.5
        elif (detections['detection_classes'][i] == 3):
            detections['detection_scores'][i] *= 5.4  # car 5.9 5.5 5.7 5.8 5.58 5.6 5.8 5.85
        elif (detections['detection_classes'][i] == 4):
            detections['detection_scores'][i] *= 7.1  # Go  3.2 3.5 3.71 3.8
        elif (detections['detection_classes'][i] == 5):
            detections['detection_scores'][i] *= 3.5  # motorcycle 4 3.8
        elif (detections['detection_classes'][i] == 6):
            detections['detection_scores'][i] *= 3.5  # pedestrian 4.5 4.6
        elif (detections['detection_classes'][i] == 7):
            detections['detection_scores'][i] *= 3  # rider
        elif (detections['detection_classes'][i] == 8):
            detections['detection_scores'][i] *= 5.8463  # Slow Down    3.5 5 6
        elif (detections['detection_classes'][i] == 9):
            detections['detection_scores'][i] *= 3.5  # Stop 3.5 3.25
        elif (detections['detection_classes'][i] == 10):
            detections['detection_scores'][i] *= 3.4  # traffic light 3.5 3.4
        elif (detections['detection_classes'][i] == 11):
            detections['detection_scores'][i] *= 5.5  # traffic sign 5 4.5
        elif (detections['detection_classes'][i] == 12):
            detections['detection_scores'][i] *= 2.38  # truck      2.2 2.3 2.4 3 2.7 2.55 2.375 2.3875 2.396 2.334


    for i in range(len(detections['detection_classes']) - 1):
        for j in range(i + 1, len(detections['detection_classes'])):
            if (detections['detection_scores'][i] >= 0.8 and detections['detection_scores'][j] >= 0.8):
                x1, y1, x2, y2 = detections['detection_boxes'][i][1], detections['detection_boxes'][i][0], \
                                 detections['detection_boxes'][i][3], detections['detection_boxes'][i][2]
                x3, y3, x4, y4 = detections['detection_boxes'][j][1], detections['detection_boxes'][j][0], \
                                 detections['detection_boxes'][j][3], detections['detection_boxes'][j][2]
                if (not ((x2 <= x3 or x4 <= x1) and (y2 <= y3 or y4 <= y1))):
                    lens = min(x2, x4) - max(x1, x3)
                    wide = min(y2, y4) - max(y1, y3)
                    areaCover = lens * wide
                    areai = areaCover / ((x2 - x1) * (y2 - y1))
                    areaj = areaCover / ((x4 - x3) * (y4 - y3))
                    if (areai >= 0.7 or areaj >= 0.7):
                        if (detections['detection_scores'][i] > detections['detection_scores'][j]):
                            detections['detection_scores'][j] = 0
                        else:
                            detections['detection_scores'][i] = 0
    for i in range(len(detections)):
        if (detections['detection_scores'][i] >= 1):
            detections['detection_scores'][i] = 0.99
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=.80,
        agnostic_mode=False)

    return image_np_with_detections

#url = 'https://www.youtube.com/watch?v=W0D2GYFCDI4'
#vPafy = pafy.new(url)
#play = vPafy.getbest(preftype="mp4")
#cap = cv2.VideoCapture(play.url)

cap = cv2.VideoCapture(0)


#cap = cv2.VideoCapture('C:/Users/Administrator.WIN-2EPKD7D6018/Desktop/amherst_train/day_test_rain_720p.mp4')
#totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#frameRate=cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#vout = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'MJPG'),frameRate, (1280,720))

prevFrame =np.array(0)
startT = time.time() * 1000.0
while(1):
    #currFrame=int((time.time() * 1000.0-startT)/1000.0*frameRate) #120
    #if currFrame >= 0 & currFrame <= totalFrames:
    if 1>0:
        # set frame position
        #cap.set(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()
        #plt.figure(figsize=(32, 18))
        #plt.imshow(currframe)
        #plt.savefig("outputSsdResnet50/test" + str(ct) + ".jpg")
        cv2.imshow("Traffic Vision", processFrame(frame))
        #vout.write(processFrame(frame))
        #cv2.imshow("Traffic Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break
    else:
        break
#cap.release()
vout.release()
cv2.destroyAllWindows()