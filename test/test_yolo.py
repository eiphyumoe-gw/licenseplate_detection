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


        checkpoint_url = "https://drive.google.com/uc?id=1bcKtEcYGIOehgPfGi_TqPkvrm6rjOUKR"
        cls.checkpoint = 'weights/small_satrn.pth'
        if not os.path.exists(os.path.dirname(cls.checkpoint)):
            os.makedirs(os.path.dirname(cls.checkpoint))
        gdown.download(checkpoint_url, cls.checkpoint, quiet=False)
        
        cls.img_1 = "../licenseplate_detection/assets/test.jpg" 
        
        config_path = '../licenseplate_detection/configs/detect.yaml'
        configs = OmegaConf.load(config_path)
        
        # device = torch.device("cuda" if configs.YOLOX.device=="gpu" else "cpu")
        device = "cpu"
        exp = get_exp(None, 'yolox-s')
        model = exp.get_model()
        model.eval()
        ckpt = torch.load(cls.weight_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        # model = fuse_model(model)
      
        cls.predictor = Predictor(
            model, exp, device= device, configs= configs, checkpoint=cls.checkpoint)

    

    def test_inference(self):
        test_image_path = self.img_1
        test_image = cv2.imread(test_image_path)

        # Perform inference on the test image
        result_img, img_info = self.predictor.inference(test_image)
        result_frame, result_str, bbox = self.predictor.visual(result_img[0], img_info, self.predictor.confthre)
        

        self.assertIsNotNone(result_frame)
        self.assertIsNotNone(result_str)
        self.assertIsNotNone(bbox)


    
    def test_removeWeight(self) -> None:
        os.remove(self.weight_path)
        self.assertFalse(os.path.exists(self.weight_path))

if __name__ == '__main__':
    unittest.main()