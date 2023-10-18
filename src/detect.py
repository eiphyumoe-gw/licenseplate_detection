import argparse
from omegaconf import OmegaConf
from utils import Demo
import json
import os

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='/home/epm/licenseplate_detection/configs/detect.yaml', help="Path to your config file")
    parser.add_argument("--path", help= "Path to your image or video")
    parser.add_argument("--output", default= '/home/epm/License_Plate_Detection/result', help= "Path to your ouput folder")
    parser.add_argument("--name", default="yolox-s", help="Please select yolox-s or yolox-l")

    return parser

def write_json(result_dict, output):
    with open(output, "w") as outfile:
        json.dump(result_dict, outfile)


def main():
    args = make_parser().parse_args()
    configs = OmegaConf.load(args.config)
    json_output = os.path.join(args.output, 'result.json')
    demo = Demo(args, configs.YOLOX)
    result_str = demo.video_demo(args.path, args.output)
    write_json(result_str, json_output)
    

if __name__ == '__main__':
    main()