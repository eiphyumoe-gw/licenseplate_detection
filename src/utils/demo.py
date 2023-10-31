import os
import cv2
import torch
import sys
sys.path.append('src')

from utils.predictor import Predictor
from YOLOX.yolox.exp import get_exp
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.utils import fuse_model
from utils import Video
from omegaconf import OmegaConf
import argparse

class Demo:
    def __init__(self, args, configs):
        self.args = args
        self.configs = configs
        self.load_model()
        self.IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
        self.predictor = Predictor(self.model, self.exp, device=self.device, configs=self.configs)
    
    def load_model(self):
        self.exp = get_exp(None, self.args.name) # select model name
        self.model = self.exp.get_model()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda" if self.configs.YOLOX.device=="gpu" else "cpu")
        if self.device == "cuda":
            self.model.cuda()
        self.model.eval()
        ckpt = torch.load(self.configs.YOLOX.ckpt, map_location="cpu")
        self.model.load_state_dict(ckpt['model'])
        self.model = fuse_model(self.model)
        
        
    def get_image_list(self, path):
        image_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext in self.IMAGE_EXT:
                    image_names.append(apath)
        return image_names


    def image_demo(self, path, output_path):
        result_dict = dict()
        final_result = list()
        if os.path.isdir(path):
            files = self.get_image_list(path)
        else:
            files = [path]
        files.sort()
        for image_name in files:
            outputs, img_info = self.predictor.inference(image_name)
            result_image, result_str, bbox = self.predictor.visual(outputs[0], img_info, self.predictor.confthre)
            result_dict['Detected_bbox'] = bbox
            result_dict['result'] = result_str
            # cv2.imshow("Test",result_frame)
            cv2.imwrite(f'{output_path}/test_{os.path.basename(image_name)}', result_image)
            final_result.append(result_dict)
        return final_result
    
    def video_demo(self, args, configs):
        final_result = list()
        frame_count = 0
        with Video(args.path, args.output,configs) as movie_loader:
            fps = movie_loader.get_fps()
            for frame in movie_loader.process_video():
                result_dict = dict()
                frame_count += 1
                with torch.no_grad():
                    result_img, img_info = self.predictor.inference(frame)
                    result_frame, result_str, bbox = self.predictor.visual(result_img[0], img_info, self.predictor.confthre)
                    result_dict['frame_id'] = frame_count
                    result_dict['current_vd_time'] = str(round((frame_count/fps), 3))+ "s"
                    result_dict['Detected_bbox'] = bbox
                    result_dict['result'] = result_str
                    # cv2.imshow("Test",result_frame)
                    final_result.append(result_dict)
                if self.configs.YOLOX.save_vd_result:
                    movie_loader.write_video(result_frame)
            movie_loader.close()
        return final_result
    
        
def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config",default="configs/detect.yaml", help = "Yaml config file")
    return parser

def main():
    video_path = '/home/epm/Downloads/Heavy Rain Japan 4K Tokyo Drive - Kayabacho to Shutoko Haneda - Downpour.mp4'
    output_path= '/home/epm/License_Plate_Detection/result/'
    args = make_parser().parse_args()
    configs = OmegaConf.load(args.config)
    d = Demo(configs.YOLOX)
    total_result_str = d.video_demo(video_path, output_path)
    print("All of the strings from license-plates are ", total_result_str)
        
if __name__ == "__main__":
    main()