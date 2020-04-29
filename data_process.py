from conf import COCO_DIR, COCO_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH
import json
from pathlib import Path
import time

class ResizeWithPad():
    def __init__(self, h, w):
        super(ResizeWithPad, self).__init__()
        self.H = IMAGE_HEIGHT
        self.W = IMAGE_WIDTH
        self.w = w
        self.h = h

    def get_transform_coefficient(self):
        if self.h <= self.w:
            longer_edge = "w"
            scale = self.W / self.w
            padding_length = (self.H - self.h * scale) / 2
        else:
            longer_edge = "h"
            scale = self.H / self.h
            padding_length = (self.W - self.w * scale) / 2
        return longer_edge, scale, padding_length

    def raw_to_resized(self, x, y, w, h):
        x = x + w / 2
        y = y + h / 2
        x = x / self.w
        y = y / self.h
        w = w / self.w
        h = h / self.h
        return x, y, w, h

    def resized_to_raw(self, center_x, center_y, width, height):
        longer_edge, scale, padding_length = self.get_transform_coefficient()
        center_x *= self.W
        width *= self.W
        center_y *= self.H
        height *= self.H
        if longer_edge == "h":
            center_x -= padding_length
        else:
            center_y -= padding_length
        center_x = center_x / scale
        center_y = center_y / scale
        width = width / scale
        height = height / scale
        return center_x, center_y, width, height


class ParseCOCO(object):
    def __init__(self):
        self.annotation_dir = COCO_DIR + "annotations/"
        self.images_dir = COCO_DIR + "train2017/"
        self.train_annotation = Path(self.annotation_dir + "instances_train2017.json")
        start_time = time.time()
        self.train_dict = self.__load_json(self.train_annotation)
        print("It took {:.2f} seconds to load the json files.".format(time.time() - start_time))
        print(self.__get_category_id_information(self.train_dict))

    def __load_json(self, json_file):
        print("Start loading {}...".format(json_file.name))
        with json_file.open(mode='r') as f:
            load_dict = json.load(f)
        print("Loading is complete!")
        return load_dict

    def __find_all(self, x, value):
        list_data = []
        for i in range(len(x)):
            if x[i] == value:
                list_data.append(i)
        return list_data

    def __get_image_information(self, data_dict):
        images = data_dict["images"]
        image_file_list = []
        image_id_list = []
        image_height_list = []
        image_width_list = []
        for image in images:
            image_file_list.append(image["file_name"])
            image_id_list.append(image["id"])
            image_height_list.append(image["height"])
            image_width_list.append(image["width"])
        return image_file_list, image_id_list, image_height_list, image_width_list

    def __get_bounding_box_information(self, data_dict):
        annotations = data_dict["annotations"]
        image_id_list = []
        bbox_list = []
        category_id_list = []
        for annotation in annotations:
            category_id_list.append(annotation["category_id"])
            image_id_list.append(annotation["image_id"])
            bbox_list.append(annotation["bbox"])
        return image_id_list, bbox_list, category_id_list

    def __get_category_id_information(self, data_dict):
        categories = data_dict["categories"]
        category_dict = {}
        for category in categories:
            category_dict[category["name"]] = category["id"]
        return category_dict

    def __process_coord(self, x, y, w, h, image_width, image_height):
        x_center, y_center, w_norm, h_norm = ResizeWithPad(h=image_height, w=image_width).raw_to_resized(x, y, w, h)
        return x_center, y_center, w_norm, h_norm

    def __bbox_information(self, image_id, image_ids_from_annotation, bboxes, image_height, image_width, category_ids):
        processed_bboxes = []
        index_list = self.__find_all(x=image_ids_from_annotation, value=image_id)
        for index in index_list:
            x, y, w, h = bboxes[index]

            x_center, y_center, w_norm, h_norm = self.__process_coord(x, y, w, h, image_width, image_height)
            processed_bboxes.append([self.__category_id_transform(category_ids[index]), x_center, y_center, w_norm, h_norm])
        return processed_bboxes

    def __category_id_transform(self, original_id):
        category_id_dict = self.__get_category_id_information(self.train_dict)
        original_name = "none"
        for category_name, category_id in category_id_dict.items():
            if category_id == original_id:
                original_name = category_name
        if original_name == "none":
            raise ValueError("An error occurred while transforming the category id.")
        return COCO_CLASSES[original_name]

    def __bbox_str(self, bboxes):
        bbox_info = ""
        for bbox in bboxes:
            for item in bbox:
                bbox_info += str(item)
                bbox_info += " "
        return bbox_info.strip()

    def write_data_to_txt(self, txt_dir):
        image_files, image_ids, image_heights, image_widths = self.__get_image_information(self.train_dict)
        image_ids_from_annotation, bboxes, category_ids = self.__get_bounding_box_information(self.train_dict)
        with open(file=txt_dir, mode="a+") as f:
            picture_index = 0
            for i in range(len(image_files)):
                write_line_start_time = time.time()
                line_info = ""
                line_info += image_files[i] + " "
                processed_bboxes = self.__bbox_information(image_ids[i],
                                                           image_ids_from_annotation,
                                                           bboxes,
                                                           image_heights[i],
                                                           image_widths[i],
                                                           category_ids)
                if processed_bboxes:
                    picture_index += 1
                    line_info += self.__bbox_str(bboxes=processed_bboxes)
                    line_info += "\n"
                    print("Writing information of the {}th picture {} to {}, which took {:.2f}s".format(picture_index, image_files[i], txt_dir, time.time() - write_line_start_time))
                    f.write(line_info)

coco = ParseCOCO()
TXT_DIR = "./data.txt"
coco.write_data_to_txt(TXT_DIR)

