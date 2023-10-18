import cv2
import os
import numpy as np
from pycocotools.coco import COCO
import json
from yolox.data.datasets import COCO_CLASSES

class COCO_Annotation(object):
    def __init__(self,coco_image_json):
        self.green = (0, 255, 0) #plate
        self.red = (255, 0, 0)  #license
        self.blue = (0, 0, 255) #car
        self.cls = COCO_CLASSES   # category names from yolox dataset
        self.color = None
        self.j_file = self.load_json(coco_image_json)
        self.coco = COCO(coco_image_json)
        
        
    def load_json(self, json_dir):
        # load json and return to this.j_file
        json_file = open(json_dir)
        self.j_file = json.load(json_file)
        return self.j_file 
        
    def load_anno(self, image_id):
        # get bounding boxes using coco
        objs = []
        anno_ids = self.coco.getAnnIds(imgIds=[int(image_id)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        for obj in annotations:
            x1 = obj['bbox'][0]
            y1 = obj['bbox'][1]
            x2 = x1 + obj['bbox'][2]
            y2 = y1 + obj['bbox'][3]
        
            print(x1, y1, x2, y2)
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                # print(obj['clean_bbox'])
                objs.append(obj)
        return objs

    def change_color(self, category):
        #change color according to category('plates', 'license', 'car')
        if category == 0:
            self.color = self.red
        elif category == 1:
            self.color = self.green
        else:
            self.color = self.blue
        return self.color

    def visualize_annotation(self, coco_annotation, coco_image_dir, val_annotation):
        # visualize bounding box on images
        for obj in coco_annotation:
            image_id = obj['image_id']
            img_filename = os.path.join(coco_image_dir, self.coco.loadImgs(image_id)[0]['file_name'])
            img = cv2.imread(img_filename)
            base_name = os.path.basename(img_filename)
            # print(base_name)
            objs = self.load_anno(image_id)
            for ix, obj in enumerate(objs):
                x1, y1, x2, y2 = obj['clean_bbox']
                category = obj['category_id']
                self.color = self.change_color(category)
                cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), self.color, 2)
                cv2.putText(img, self.cls[category], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color, thickness=1)
            cv2.imwrite(f'{val_annotation}/'+ base_name , img)

def main():
    coco_image_json = r'/home/epm/YOLOX/datasets/COCO/annotations/instances_train2017.json'
    coco_image_dir = r'/home/epm/YOLOX/datasets/COCO/train2017'
    val_annotation = r'/home/epm/YOLOX/datasets/COCO/train_annotation'

    if not os.path.exists(val_annotation):
        os.mkdir(val_annotation)

    annotator = COCO_Annotation(coco_image_json) 
    j_file = annotator.load_json(coco_image_json)
    coco_anno = j_file['annotations']
    annotator.visualize_annotation(coco_anno, coco_image_dir, val_annotation)
    
if __name__ == "__main__":
    main()