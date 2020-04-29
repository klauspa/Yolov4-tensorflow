"""
mosaic data argumentation tensorflow implementation
reference: https://github.com/clovaai/CutMix-PyTorch https://github.com/AlexeyAB/darknet
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import cv2
from read_txt import ReadTxt
import os
import random
from conf import COCO_DIR, TRAIN_DIR, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS, DATA_ARG_FACTOR
TXT_DIR = "./data.txt"
BATCH_SIZE = 4
data_factors = DATA_ARG_FACTOR()


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

#load_classification_data()

def random_gen():
    return np.random.randint(10000)

def rand_int(min, max):
    if max < min:
        min, max = max, min

    r = (random_gen()%(max - min + 1)) + min
    return r

def random_float():
    return np.random.rand()

def rand_uniform_strong(min, max):
    if (max < min):
        min, max = max, min
    return (random_float() * (max - min)) + min

def rand_scale(s):
    scale = rand_uniform_strong(1, s)
    if(random_gen()%2):
        return scale
    return 1./scale

def draw_boxes(images, boxes):
    for i in range(BATCH_SIZE):
        img = images[i].numpy()
        cv2.imwrite("hello.jpg", img)
        img = cv2.imread("hello.jpg")
        for j in range(len(boxes[i])):
            x = boxes[i][j][1]
            y = boxes[i][j][2]
            w = boxes[i][j][3]
            h = boxes[i][j][4]

            left = int((x - w / 2) * IMAGE_WIDTH)
            top = int((y - h / 2) * IMAGE_HEIGHT)
            right = int((x + w / 2) * IMAGE_WIDTH)
            bot = int((y + h / 2) * IMAGE_HEIGHT)

            cv2.rectangle(img, (left, top), (right, bot), (0,0,255), 2)
        cv2.resize(img,(224, 224))
        cv2.imwrite(str(i)+".jpg", img)


def load_img(file_path):

    img_raw = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(img_raw, channels=CHANNELS)
    image = tf.image.adjust_saturation(image, rand_scale(data_factors.saturation))
    image = tf.image.adjust_hue(image, rand_uniform_strong(-1*data_factors.hue, data_factors.hue))
    image = tf.image.adjust_contrast(image, rand_scale(data_factors.exposure))
    #image = tf.image.resize_with_pad(image=image, target_height=IMAGE_HEIGHT, target_width=IMAGE_WIDTH)
    image = tf.image.resize(images=image, size=(IMAGE_HEIGHT,IMAGE_WIDTH))

    return image

def merge_bboxes(bboxes, cutx, cuty):
    cutx = cutx / IMAGE_WIDTH
    cuty = cuty / IMAGE_HEIGHT

    merge_bbox = []
    for i in range(bboxes.shape[0]):
        for box in bboxes[i]:
            tmp_box = []
            x,y,w,h = box[1], box[2], box[3], box[4]

            if i == 0:
                if box[2]-box[4]/2 > cuty or box[1]-box[3]/2 > cutx:
                    continue

                if box[2]+box[4]/2 > cuty and box[2]-box[4]/2 < cuty:
                    h -= (box[2]+box[4]/2-cuty)
                    y -= (box[2]+box[4]/2-cuty)/2

                if box[1]+box[3]/2 > cutx and box[1]-box[3]/2 < cutx:
                    w -= (box[1]+box[3]/2-cutx)
                    x -= (box[1]+box[3]/2-cutx)/2
                
            if i == 1:
                if box[2]+box[4]/2 < cuty or box[1]-box[3]/2 > cutx:
                    continue

                if box[2]+box[4]/2 > cutx and box[2]-box[4]/2 < cutx:
                    h -= (cuty-(box[2]-box[4]/2))
                    y += (cuty-(box[2]-box[4]/2))/2
                
                if box[1]+box[3]/2 > cutx and box[1]-box[3]/2 < cutx:
                    w -= (box[1]+box[3]/2-cutx)
                    x -= (box[1]+box[3]/2-cutx)/2

            if i == 2:
                if box[2]+box[4]/2 < cuty or box[1]+box[3]/2 < cutx:
                    continue

                if box[2]+box[4]/2 < 1 and box[2]-box[4]/2 < cuty:
                    h -= (cuty-(box[2]-box[4]/2))
                    y += (cuty-(box[2]-box[4]/2))/2

                if box[1]+box[3]/2 > cutx and box[1]-box[3]/2 < cutx:
                    w -= (cutx-(box[1]-box[3]/2))
                    x += (cutx-(box[1]-box[3]/2))/2

            if i == 3:
                if box[2]-box[4]/2 > cuty or box[1]+box[3]/2 < cutx:
                    continue

                if box[2]+box[4]/2 > cuty and box[2]-box[4]/2 < cuty:
                    h -= (box[2]+box[4]/2-cuty)
                    y -= (box[2]+box[4]/2-cuty)/2

                if box[1]+box[3]/2 > cutx and box[1]-box[3]/2 < cutx:
                    w -= (cutx-(box[1]-box[3]/2))
                    x += (cutx-(box[1]-box[3]/2))/2

            tmp_box.append(box[0])
            tmp_box.append(x)
            tmp_box.append(y)
            tmp_box.append(w)
            tmp_box.append(h)
            merge_bbox.append(tmp_box)
            
    #TO DO:eliminate small boxes
    #may be no boxes

    if len(merge_bbox) == 0:
        return None
    else:
        return merge_bbox

def mosaic_process(image_batch, label_batch):
    """default dataset: coco
       mosaic data argumentation
    >args
    -------

    """
    #usr_mix = 0 no mosaic use_mix = 3 use mosaic
    
    use_mix = 3
    #num of image
    n = len(image_batch)

    cut_x, cut_y = [0]*n, [0]*n
    random_index = random_gen()
    #if (random_index % 2 == 0): use_mix = 1
    if (use_mix == 3):
        min_offset = 0.2
        for i in range(n):
            h = IMAGE_HEIGHT
            w = IMAGE_WIDTH
            cut_x[i] = np.random.randint(int(w*min_offset), int(w*(1 - min_offset)))
            cut_y[i] = np.random.randint(int(h*min_offset), int(h*(1 - min_offset)))
            #cut_x[i] = random.uniform(min_offset, (1-min_offset))
            #cut_y[i] = random.uniform(min_offset, (1-min_offset))

    augmentation_calculated, gaussian_noise = 0, 0

    def get_random_paths():
        random_index = random.sample(list(range(n)), use_mix+1)

        random_paths = []
        random_bboxes = []
        for idx in random_index:
            random_paths.append(os.path.join(COCO_DIR, TRAIN_DIR, image_batch[idx]))
            random_bboxes.append(label_batch[idx])
        return random_paths, np.array(random_bboxes)

    #n images per batch, we also generate n images if mosaic
    
    if (use_mix == 3):

        dest = []
        new_boxes = []
        for i in range(n):
            paths, bboxes = get_random_paths()
            img0 = load_img(paths[0])
            img1 = load_img(paths[1])
            img2 = load_img(paths[2])
            img3 = load_img(paths[3])

            #cut and adjust
            d1 = img0[:cut_y[i], :cut_x[i], :]
            d2 = img1[cut_y[i]:, :cut_x[i], :]
            d3 = img2[cut_y[i]:, cut_x[i]:, :]
            d4 = img3[:cut_y[i], cut_x[i]:, :]

            tmp1 = tf.concat([d1, d2], axis=0)
            tmp2 = tf.concat([d4, d3], axis=0)

            dest.append(tf.concat([tmp1, tmp2], axis=1))
            #print(bboxes)

            tmp_boxes = (merge_bboxes(bboxes, cut_x[i], cut_y[i]))
            if not tmp_boxes:
                i = i - 1
                continue
            new_boxes.append(tmp_boxes)
            
        dest = tf.stack(dest)
        
        draw_boxes(dest, new_boxes)
        return dest, new_boxes
        
    
    if (use_mix == 0):
        dest = tf.zeros([n, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS])
        for i in range(n):
            paths, bboxes = get_random_paths()
            dest[i] = load_img(paths[0])
        new_boxes = label_batch

        return dest, new_boxes


def get_length_of_dataset(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count

def generate_dataset():
    txt_dataset = tf.data.TextLineDataset(filenames=TXT_DIR)
    train_count = get_length_of_dataset(txt_dataset)
    train_dataset = txt_dataset.batch(batch_size=BATCH_SIZE)

    return train_dataset, train_count

def parse_dataset_batch(dataset):
    """
     Return :
     image_name_list : list, length is N (N is the batch size.)
     boxes_array : numpy.ndarrray, shape is (N, MAX_TRUE_BOX_NUM_PER_IMG, 5)
    """
    image_name_list = []
    boxes_list = []
    len_of_batch = dataset.shape[0]
    for i in range(len_of_batch):
        image_name, boxes = ReadTxt(line_bytes=dataset[i].numpy()).parse_line()
        image_name_list.append(image_name)
        boxes_list.append(boxes)
    boxes_array = np.array(boxes_list)
    return image_name_list, boxes_array

if __name__ == "__main__":
    #get txt dataset which contains filename、boexs、label in text format
    train_dataset, train_count = generate_dataset()

    step = 0
    for dataset_batch in train_dataset:
        step += 1
        images, boxes = parse_dataset_batch(dataset=dataset_batch)
        
        images, boxes = mosaic_process(images, boxes)
        print(images.shape)

        #draw_boxes(images, boxes)
