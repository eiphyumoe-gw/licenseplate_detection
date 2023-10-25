import os
import unittest
import gdown
import cv2
import torch
import sys
# from loguru import logger
sys.path.append('src')

from YOLOX.yolox.exp import get_exp
from YOLOX.yolox.data.datasets import COCO_CLASSES
from utils.predictor import Predictor
from YOLOX.yolox.utils import fuse_model
from omegaconf import OmegaConf

class test_yoloxpredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        yolox_s_url = "https://drive.google.com/uc?id=1IIHvFx0Aby71ux_E5idH6ssfBbSXa0Tw"
        cls.weight_path = 'weights/yolox_s.pth'
        if not os.path.exists(os.path.dirname(cls.weight_path)):
            os.makedirs(os.path.dirname(cls.weight_path))
        gdown.download(yolox_s_url, cls.weight_path, quiet = False)

        cls.img = "../licenseplate_detection/assets/test.jpg" 
        
        config_path = '../licenseplate_detection/configs/detect.yaml'
        configs = OmegaConf.load(config_path)
        
        device = torch.device("cuda" if configs.YOLOX.device=="gpu" else "cpu")
        print("Device is ", device)
        exp = get_exp(None, 'yolox-s')
        model = exp.get_model()
        model.cuda()
        model.eval()
        cls_names = COCO_CLASSES
        trt_file = None
        decoder = None
        ckpt = torch.load(configs.YOLOX.ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model = fuse_model(model)
      
        cls.predictor = Predictor(
            model, exp, device= device, configs= configs)


    def test_inference(self):
        test_image_path = self.img
        test_image = cv2.imread(test_image_path)

        # Perform inference on the test image
        outputs, img_info = self.predictor.inference(test_image)

        self.assertIsNotNone(outputs)
        self.assertIsNotNone(img_info)


    # def test_removeFile(self) -> None:
    #     os.remove(self.img)
    #     self.assertFalse(os.path.exists(self.img))
    
    def test_removeWeight(self) -> None:
        os.remove(self.weight_path)
        self.assertFalse(os.path.exists(self.weight_path))


class cfg():

    #constructor
    def __init__(self, **dict):
        self.__dict__.update(dict)


if __name__ == '__main__':
    unittest.main()