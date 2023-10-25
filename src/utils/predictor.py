import sys 
sys.path.append('src')

from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.data.data_augment import ValTransform
from YOLOX.yolox.utils import postprocess, vis
from OCR.vedastr.utils import Config
from OCR.vedastr.runners import InferenceRunner
import cv2
import os
import torch
import time
from loguru import logger

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cuda",
        fp16=False,
        legacy=False,
        configs = None,
        checkpoint = None,
    ):
        
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.configs = configs
        self.num_classes = len(cls_names)
        self.confthre = self.configs.YOLOX.conf
        self.nmsthre = self.configs.YOLOX.nms
        self.checkpoint = self.configs.OCR.checkpoint
        self.test_size = (self.configs.YOLOX.tsize,self.configs.YOLOX.tsize)
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.load_inference_runner()
    
        # if trt_file is not None:
        #     from torch2trt import TRTModule

        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))

        #     x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
        #     self.model(x)
        #     self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)

        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "cuda":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img.cuda())
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info
    
    def visual(self, output, img_info, cls_conf=0.35):
        total_result_str = list()
        bbox_count = 0
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img, total_result_str, bbox_count
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        for i in range(len(bboxes)):
            img_cpy = img.copy()
            box = bboxes[i]
            cls_id = int(cls[i])
            score = scores[i]
            if cls_id == 1 and score > cls_conf:
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                bbox_count += 1
                
                x0 = x0 if x0>-1 else 0
                x1 = x1 if x1>-1 else 0
                y0 = y0 if y0>-1 else 0
                y1 = y1 if y1>-1 else 0
                img_cpy = img_cpy[y0:y1, x0:x1]            
                result_str = self.read_license_from_image(img_cpy)
                total_result_str.append(result_str)
                # cv2.imwrite(f'result/test_{i}.png', img_cpy)
    
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res, total_result_str, bbox_count
    
    def read_license_from_image(self, image):
        """ This function return predicted string of the text on the license plate.

        Args:
            image : Cropped image of only license plate

        Returns:
            str: strings predicted from Scene Text Recognition (small_satrn model)
        """
        
        pred_str, probs = self.runner(image)
        # self.runner.logger.info('predict string: {}'.format(pred_str))
        return pred_str
    
    
    def load_inference_runner(self):
        cfg_path = self.configs.OCR.cfg
        cfg = Config.fromfile(cfg_path)
        deploy_cfg = cfg['deploy']
        common_cfg = cfg.get('common')
        deploy_cfg['device'] = self.configs.OCR.device
        
        if deploy_cfg['device'] == 'gpu':
            deploy_cfg['gpu_id'] = self.configs.OCR.gpus
        else:
            deploy_cfg['gpu_id'] = None
        self.runner = InferenceRunner(deploy_cfg, common_cfg)
        self.runner.load_checkpoint(self.checkpoint)

    
