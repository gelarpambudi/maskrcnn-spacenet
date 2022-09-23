import os
import json
import skimage
import numpy as np
import sys

sys.path.append("/path/to/mask/rcnn/repo")
from mrcnn import utils


class Spacenet_dataset(utils.Dataset):

    def load_dataset(self, dataset_dir, dataset_type):
        """
        Generate train and validation dataset
        """
        self.add_class("building", 1, "building")

        data_dir = os.path.join(dataset_dir, dataset_type)
        annotations_file = os.path.join(data_dir, "building_annotation.json")

        #load annotation file
        json_data = json.load(open(annotations_file))
        json_data_value = list(json_data.values())
        annotations = [ x for x in json_data_value if x['regions']]

        for annot in annotations:
            if type(annot['regions']) is dict:
                polygons = [ r['shape_attributes'] for r in annot['regions'].values() ]
            else:
                polygons = [ r['shape_attributes'] for r in annot['regions'] ]
                

            img_file_path = os.path.join(data_dir, annot['filename'])
            img = skimage.io.imread(img_file_path)
            img_height, img_width = img.shape[:2]

            self.add_image(
                "building",
                image_id=annot['filename'],
                path=img_file_path,
                width=img_width,
                height=img_height,
                polygons=polygons
            )


    def load_mask(self, image_id):
        """
        Load image mask
        """
        img_info = self.image_info[image_id]
        if img_info["source"] != "building":
            return super(self.__class__, self).load_mask(image_id)

        #draw masks
        img_height = img_info["height"] 
        img_width = img_info["width"]
        polygon_length = len(img_info["polygons"])

        mask = np.zeros([img_height, img_width, polygon_length], dtype=np.uint8)
        for i, p in enumerate(img_info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        
        class_ids = np.ones([mask.shape[-1]])
        return mask.astype(bool), class_ids


    def image_reference(self, image_id):
        """
        Get path of the image
        """        
        img_info = self.image_info[image_id]
        if img_info["source"] == "building":
            return img_info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)
