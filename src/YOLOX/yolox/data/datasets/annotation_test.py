import cv2
import os
import json
from collections import defaultdict

class COCO_Annotation(object):
    def __init__(self):
        self.cls = {0: "plate", 1: "license", 2 : "car"}   # category names 
        self.color = {"plate": (0 ,255, 0),
                      "license": (255, 225, 0),
                      "car": (225, 0, 255)}
        self.j_file =  None
        
        
    def load_json(self, json_dir):
        # load json and return to this.j_file
        with open(json_dir, 'r') as file:
            self.j_file = json.load(file)
        return self.j_file
    
    def getAnnIds(self, json_file):
        img_anno = defaultdict(list)
        for obj in json_file['annotations']:
            img_anno[obj['image_id']].append(obj['id'])
        return img_anno
    
    def loadAnns(self, img_anno_ids):
        annotation = list()
        for id in img_anno_ids:
            annotation.append(self.j_file['annotations'][id])
        return annotation
    
    def visualize_annotation(self, coco_images, coco_image_dir, val_annotation):
        # visualize bounding box on images
        img_anno_ids = self.getAnnIds(self.j_file)
        for coco_image in coco_images:
            image_id = coco_image['id']
            anno_id = img_anno_ids[image_id]
            img_name = coco_image['file_name']
            img_filename = os.path.join(coco_image_dir, img_name)
            img = cv2.imread(img_filename)

            annotations = self.loadAnns(anno_id)
            # print("+++annotation+++", len(annotations))
            for obj in annotations:
                x1 = obj['bbox'][0]
                y1 = obj['bbox'][1]
                x2 = x1 + obj['bbox'][2]
                y2 = y1 + obj['bbox'][3]
                category = obj['category_id']
                class_name = self.cls[category]
                cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), self.color[class_name], 2)
                cv2.putText(img, class_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color[class_name], thickness=1)
            cv2.imwrite(f'{val_annotation}/{img_name}' , img)
        print("Done")
def main():
    coco_image_json = r'/home/epm/YOLOX/datasets/COCO/annotations/instances_val2017.json'
    coco_image_dir = r'/home/epm/YOLOX/datasets/COCO/val2017'
    val_annotation = r'/home/epm/YOLOX/datasets/COCO/val_annotation'

    if not os.path.exists(val_annotation):
        os.mkdir(val_annotation)

    annotator = COCO_Annotation() 
    j_file = annotator.load_json(coco_image_json)
    coco_images = j_file['images']
    annotator.visualize_annotation(coco_images, coco_image_dir, val_annotation)
    
if __name__ == "__main__":
    main()