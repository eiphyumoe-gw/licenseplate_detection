import cv2
import os
import numpy as np
from pycocotools.coco import COCO
import json

class COCO_Annotation(object):
    def __init__(self,coco_image_json):
        self.cls = {0: "plate", 1: "license", 2 : "car"}   # category names 
        self.color = {"plate": (0 ,255, 0),
                      "license": (255, 225, 0),
                      "car": (225, 0, 255)}
        self.j_file =  None
        self.coco = COCO(coco_image_json)
        
    def load_json(self, json_dir):
        # load json and return to this.j_file
        with open(json_dir, 'r') as file:
            self.j_file = json.load(file)
        return self.j_file
        
    def visualize_annotation(self, coco_annotation, coco_image_dir, val_annotation):
        # visualize bounding box on images
        for obj in coco_annotation:
            image_id = obj['image_id']
            img_filename = os.path.join(coco_image_dir, self.coco.loadImgs(image_id)[0]['file_name'])
            img = cv2.imread(img_filename)
            base_name = os.path.basename(img_filename)
            # print(base_name)
            
            anno_ids = self.coco.getAnnIds(imgIds=[int(image_id)], iscrowd=False)
            annotations = self.coco.loadAnns(anno_ids)
            for obj in annotations:
                x1 = obj['bbox'][0]
                y1 = obj['bbox'][1]
                x2 = x1 + obj['bbox'][2]
                y2 = y1 + obj['bbox'][3]
                category = obj['category_id']
                class_name = self.cls[category]
                cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), self.color[class_name], 2)
                cv2.putText(img, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color[class_name], thickness=1)
            cv2.imwrite(f'{val_annotation}/'+ base_name , img)
        print("Done")
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