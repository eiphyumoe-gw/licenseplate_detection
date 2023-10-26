'''
Filename: ../licenseplate_detection/src/detect.py
Path: ../licenseplate_detection/src
Created Date: Wednesday, October 18th 2023, 4:16:08 pm
Author: EPM

Copyright (c) 2023 GlobalWalkers,Inc. All rights reserved.

'''

import argparse
from omegaconf import OmegaConf
from utils import Demo
import json
import os
import time

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='../licenseplate_detection/configs/detect.yaml', help="Path to your config file")
    parser.add_argument("--demo", default="video", help="demo type: image, video")
    parser.add_argument("--path", required=True, help= "Path to your image or video")
    parser.add_argument("--output", default= '../licenseplate_detection/result', help= "Path to your ouput folder")
    parser.add_argument("--name", default="yolox-s", help="Please select yolox-s or yolox-l")
    return parser

def write_json(result_str, output):
    with open(output, "w") as outfile:
        json.dump(result_str, outfile)


def main():
    args = make_parser().parse_args()
    configs = OmegaConf.load(args.config)
    os.makedirs(args.output, exist_ok=True)
    assert os.path.exists(args.path), "Input Path Error"

    json_output = os.path.join(args.output, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + ".json")
    demo = Demo(args, configs)
    if args.demo == "video":
        result_str = demo.video_demo(args, configs)
        write_json(result_str, json_output)
    else:
        result_str = demo.image_demo(args.path, args.output)
        print(result_str)

    

if __name__ == '__main__':
    main()