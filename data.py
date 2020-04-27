"""
mosaic data argumentation tensorflow implementation
reference: https://github.com/clovaai/CutMix-PyTorch https://github.com/AlexeyAB/darknet
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import cv2

parser = argparse.ArgumentParser(description="mosaic data argumentation tensorflow implementation")
parser.add_argument("--path", default="./imagenet_test", type=str)
args = parser.parse_args()

def load_classification_data():
    """
    two classes imagenet_test data folder as a test
    """
    train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
    train_data_gen = train_image_generator.flow_from_directory(batch_size=4,
                                                           directory=args.path,
                                                           shuffle=True,
                                                           target_size=(224, 224),
                                                           class_mode='binary')
    steps = 4
    while (steps > 0):
        for inputs, target in train_data_gen:
            min_offset = 0.2
            w = inputs.shape[1]
            h = inputs.shape[2]
            cut_x = np.random.randint(int(w*min_offset), int(w*(1 - min_offset)))
            cut_y = np.random.randint(int(h*min_offset), int(h*(1 - min_offset)))

            s1 = (cut_x * cut_y) // (w*h)
            s2 = ((w - cut_x) * cut_y) // (w*h)
            s3 = (cut_x * (h - cut_y)) // (w*h)
            s4 = ((w - cut_x) * (h - cut_y)) // (w*h)

            d1 = inputs[0, :(h-cut_y), 0:cut_x, :]
            d2 = inputs[1, (h-cut_y):, 0:cut_x, :]
            d3 = inputs[2, (h-cut_y):, cut_x:, :]
            d4 = inputs[3, :(h-cut_y), cut_x:, :]

            tmp1 = np.vstack((d1, d2))
            tmp2 = np.vstack((d4, d3))

            tmpx = np.hstack((tmp1, tmp2))
            tmpx = tmpx*255
            tmpy = target[0]*s1 + target[1]*s2 + target[2]*s3 + target[3]*s4

            cv2.imwrite("argumentation.jpg", tmpx)



            break
        
        steps -= 1

load_classification_data()

    


#mosaic data argumentation for detection data
def blend_truth_mosaic(new_truth, boxes, old_truth, w, h, cut_x, cut_y, i_mixup,
                        left_shift, right_shift, top_shift, bot_shift):
    """
    not yet finished
    """
    t_size = 4+1
    count_new_truth = 0;
    for t in range(boxes):
        try:
            x = new_truth[t*(4+1)]
        except:
            break;
        count_new_truth += 1
    new_t = count_new_truth

    for t in range(count_new_truth, boxes):
        new_truth_ptr = new_truth[new_t*t_size:]
        new_truth_ptr[0] = 0
        old_truth_ptr = old_truth[(t - count_new_truth)*t_size:]
        try:
            x = new_truth_ptr[0]
        except:
            break;
        
    xb = old_truth_ptr[0]
    yb = old_truth_ptr[1]
    wb = old_truth_ptr[2]
    hb = old_truth_ptr[3]

    #shift 4 images
    if (i_mixup == 0):
        xb = xb - (w - cut_x - right_shift) / w
        yb = yb - (h - cut_y - bot_shift) / h
    if (i_mixup == 1):
        xb = xb + (cut_x - left_shift) / w
        yb = yb - (h - cut_y - bot_shift) / h
    if (i_mixup == 2):
        xb = xb - (w - cut_x - right_shift) / w
        yb = yb + (cut_y - top_shift) / h
    if (i_mixup == 3):
        xb = xb + (cut_x - left_shift) / w
        yb = yb + (cut_y - top_shift) / h

    left = (xb - wb / 2)*w
    right = (xb + wb / 2)*w
    top = (yb - hb / 2)*h
    bot = (yb + hb / 2)*h

    # fix out of bound
    if (left < 0):
        diff = left / w
        xb = xb - diff // 2
        wb = wb + diff

    if (right > w):
        diff = (right - w) / w
        xb = xb - diff // 2
        wb = wb - diff

    if (top < 0):
        diff = top / h
        yb = yb - diff // 2
        hb = hb + diff

    if (bot > h):
        diff = (bot - h) / h
        yb = yb - diff // 2
        hb = hb - diff

    left = (xb - wb // 2)*w
    right = (xb + wb // 2)*w
    top = (yb - hb // 2)*h
    bot = (yb + hb // 2)*h

    # leave only within the image
    if(left >= 0 and right <= w and top >= 0 and bot <= h and
        wb > 0 and wb < 1 and hb > 0 and hb < 1 and
        xb > 0 and xb < 1 and yb > 0 and yb < 1):

        new_truth_ptr[0] = xb
        new_truth_ptr[1] = yb
        new_truth_ptr[2] = wb
        new_truth_ptr[3] = hb
        new_truth_ptr[4] = old_truth_ptr[4]
        new_t += 1

