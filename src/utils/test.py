from predictor import Predictor
import os
import cv2

import sys
sys.path.append('src')

from YOLOX.yolox.exp import get_exp
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.utils import fuse_model
import torch

class Demo(Predictor):
    def __init__(self):
        ckpt_file = '/home/epm/LicensePlate_Detection/YOLOX/YOLOX_outputs/yolox_s/best_ckpt.pth'
        ckpt = torch.load(ckpt_file, map_location="cpu")
        exp = get_exp(None, 'yolox-s') # select model name
        model = exp.get_model()
        model.cuda()
        model.eval()
        model.load_state_dict(ckpt['model'])
        model = fuse_model(model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
        self.predictor = Predictor(model, COCO_CLASSES, device)
        
        
    def get_image_list(self, path):
        image_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in self.IMAGE_EXT:
                    image_names.append(apath)
        return image_names



    def image_demo(self, path):
        if os.path.isdir(path):
            files = self.get_image_list(path)
        else:
            files = [path]
        files.sort()
        for image_name in files:
            outputs, img_info = self.predictor.inference(image_name)
            result_image, result_str = self.predictor.visual(outputs[0], img_info, self.predictor.confthre)
            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        return result_image, result_str
    
    
def main():
    img_path = r'/home/epm/License_Plate_Detection/test.jpg'
    d = Demo()
    result_image, total_result_str = d.image_demo(img_path)
    cv2.imwrite(r'/home/epm/License_Plate_Detection/result/result.png', result_image)
    print("All of the strings from license-plates are ", total_result_str)
        
if __name__ == "__main__":
    main()