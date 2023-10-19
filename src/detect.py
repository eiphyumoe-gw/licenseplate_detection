import argparse
from omegaconf import OmegaConf
from utils import Demo
import json
import os

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='/home/epm/licenseplate_detection/configs/detect.yaml', help="Path to your config file")
    parser.add_argument("--path", help= "Path to your image or video")
    parser.add_argument("--output", default= '/home/epm/licenseplate_detection/result', help= "Path to your ouput folder")
    parser.add_argument("--name", default="yolox-s", help="Please select yolox-s or yolox-l")

    return parser

def write_json(result_str, output):
    with open(output, "w") as outfile:
        json.dump(result_str, outfile)


def main():
    args = make_parser().parse_args()
    configs = OmegaConf.load(args.config)
    json_output = os.path.join(args.output, 'predicted.json')
    demo = Demo(args, configs.YOLOX)
    result_str = demo.video_demo(args.path, args.output)
    print("Output is ", json_output)
    write_json(result_str, json_output)
    

if __name__ == '__main__':
    main()