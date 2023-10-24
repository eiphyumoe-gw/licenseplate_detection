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


class test_yoloxpredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        yolox_nano_url = "https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_s.py"
        cls.weight_path = 'weights/yolox_s.pth'
        if not os.path.exists(os.path.dirname(cls.weight_path)):
            os.makedirs(os.path.dirname(cls.weight_path))
        gdown.download(yolox_nano_url, cls.weight_path, quiet = False)

        sample_img_url = "../licenseplate_detection/assets/test.jpg" 
        cls.img = 'test_data/test.jpg'
        if not os.path.exists(os.path.dirname(cls.img)):
            os.makedirs(os.path.dirname(cls.img))

        yolox_cfg = {
            'conf_thres': 0.7,
            'nms': 0.45,
            'tsize': 640,
            'device': 'cpu',
            'fp16': False,
            'legacy': False,
            'weight_path': cls.weight_path
        }
        yolox_cfg = cfg(**yolox_cfg)
        exp = get_exp(None, 'yolox_s')
        model = exp.get_model()
        model.cuda()
        model.eval()
        cls_names = COCO_CLASSES
        trt_file = None
        decoder = None

        ckpt = torch.load(cls.weight_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model = fuse_model(model)
        
        cls.predictor = Predictor(
            model, exp, cls_names, trt_file, decoder, yolox_cfg.device, yolox_cfg.fp16, yolox_cfg.legacy, yolox_cfg)


    def test_inference(self):
        test_image_path = self.img
        test_image = cv2.imread(test_image_path)

        # Perform inference on the test image
        outputs, img_info = self.predictor.inference(test_image)

        self.assertIsNotNone(outputs)
        self.assertIsNotNone(img_info)


    def test_removeFile(self) -> None:
        os.remove(self.img)
        self.assertFalse(os.path.exists(self.img))
    
    def test_removeWeight(self) -> None:
        os.remove(self.weight_path)
        self.assertFalse(os.path.exists(self.weight_path))


class cfg():

    #constructor
    def __init__(self, **dict):
        self.__dict__.update(dict)


if __name__ == '__main__':
    unittest.main()