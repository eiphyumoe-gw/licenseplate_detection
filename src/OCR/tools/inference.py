import argparse
import os
import sys
from time import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2

from OCR.vedastr.runners import InferenceRunner
from OCR.vedastr.utils import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('config', type=str, help='Config file path')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file path')
    parser.add_argument('image', type=str, help='input image path')
    parser.add_argument('gpus', type=str, help='target gpus')
    args = parser.parse_args()

    return args


def main():
    end_time = 0
    args = parse_args()

    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    deploy_cfg = cfg['deploy']
    common_cfg = cfg.get('common')
    deploy_cfg['gpu_id'] = args.gpus.replace(" ", "")

    runner = InferenceRunner(deploy_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)
    if os.path.isfile(args.image):
        images = [args.image]
    else:
        images = [os.path.join(args.image, name)
                  for name in os.listdir(args.image)]
    for img in images:
        assert os.path.exists(img), f"{img} not exists"
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start_time = time()
        pred_str, probs = runner(image)
        end_time = time() - start_time
        runner.logger.info('predict string: {} \t of {}'.format(pred_str, img))
        runner.logger.info('FPS {}'.format((1/ end_time)))

if __name__ == '__main__':
    main()
