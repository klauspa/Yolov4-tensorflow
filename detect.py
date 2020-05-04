import tensorflow as tf 
import numpy as np
from conf import ANCHORS, COCO_CLASSES, XYSCALE, IMAGE_HEIGHT, IMAGE_WIDTH, CATEGORY_NUM, STRIDES
from model_infer import Yolo_Model, load_weights
from prepost_process import postprocess_boxes, postprocess_bbbox, nms, draw_bbox, image_preporcess
import argparse
import cv2
import time

parser = argparse.ArgumentParser(description='yolov4 detect args')
parser.add_argument('--image', type=str)
parser.add_argument('--weight', type=str, default='yolov4.weights')
args = parser.parse_args()

if __name__ == "__main__":
    anchors = np.array(ANCHORS)
    anchors = np.reshape(anchors, [3, 3, 2])
    num_classes = len(COCO_CLASSES)
    xy_scale = XYSCALE
    input_size = IMAGE_WIDTH

    #input image path
    image_path = args.image
    img = cv2.imread(image_path)
    original_image = img
    original_image_size = img.shape[:2]
    image_data = image_preporcess(np.copy(original_image), [input_size, input_size])

    img_tensor = tf.convert_to_tensor(image_data, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, axis=0)

    time_p1 = time.time()
    model = Yolo_Model()
    
    load_weights(model, args.weight)
    time_p2 = time.time()

    pred_bbox = model.predict(img_tensor)
    time_p3 = time.time()

    pred_bbox = postprocess_bbbox(pred_bbox, anchors, STRIDES, XYSCALE)
    bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
    bboxes = nms(bboxes, 0.213, method='nms')
    time_p4 = time.time()

    image = draw_bbox(original_image, bboxes)
    cv2.imwrite("detect.jpg", image)

    print("load model: ", time_p2-time_p1)
    print("forward: ", time_p3-time_p2)
    print("post process: ", time_p4-time_p3)

