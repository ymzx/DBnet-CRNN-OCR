# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import threading

__dir__ = os.path.dirname(__file__)
sys.path.append(os.path.join(__dir__, ''))

import cv2
import numpy as np
from pathlib import Path
import requests
from tqdm import tqdm

from configs.config import args, A4
from tools.infer import predict_system
from ppocr.utils.log import get_logger
from tools.utils.simple_funcs import ocr_format_convert, restore_img_size, custom_img_resize

logger = get_logger()
from ppocr.utils.utility import check_and_read_gif, get_image_file_list

__all__ = ['PaddleOCR']


SUPPORT_DET_MODEL = ['DB']
VERSION = 2.0
SUPPORT_REC_MODEL = ['CRNN']
BASE_DIR = os.path.expanduser("~/.paddleocr/")


def download_with_progressbar(url, save_path):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes == 0 or progress_bar.n != total_size_in_bytes:
        logger.error("Something went wrong while downloading models")
        sys.exit(0)



class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        self.use_angle_cls = args.use_angle_cls
        project_path =  Path(__file__).parent # 项目绝对路径
        args.rec_char_dict_path = project_path / args.rec_char_dict_path
        # init det_model and rec_model
        super().__init__(args)

    def ocr(self, img, det=True, rec=True, cls=False):
        self.true_ = """
        ocr with paddleocr
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not, if false, only rec will be exec. default is True
            rec: use text recognition or not, if false, only det will be exec. default is True
        """
        self.use_angle_cls = cls
        scale = 1.0
        if isinstance(img, str):
            image_file = img
            img, flag = check_and_read_gif(image_file)
            if not flag:
                with open(image_file, 'rb') as f:
                    np_arr = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                logger.error("error in loading image:{}".format(image_file))
                return None
        if isinstance(img, np.ndarray) and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if A4: img, scale = custom_img_resize(img)
        logger.info("input image size (%d,%d,%d)" % img.shape)
        if det and rec:
            dt_boxes, rec_res, chars_pos, chars_score = self.__call__(img)
            if dt_boxes is None: return None
            dt_boxes, chars_pos = restore_img_size(dt_boxes=dt_boxes, chars_box=chars_pos, scale=scale)
            return [[box.tolist(), res, c_pos, c_score] for box, res, c_pos, c_score in zip(dt_boxes, rec_res, chars_pos, chars_score)]
        elif det and not rec:
            dt_boxes, elapse = self.text_detector(img)
            dt_boxes = restore_img_size(dt_boxes=dt_boxes, scale=scale)
            if dt_boxes is None: return None
            return [box.tolist() for box in dt_boxes]
        else:
            if not isinstance(img, list):
                img = [img]
            if self.use_angle_cls:
                img, cls_res, elapse = self.text_classifier(img)
                if not rec: return cls_res
            rec_res, chars_pos, chars_score, elapse = self.text_recognizer(img)
            chars_pos = restore_img_size(chars_box=chars_pos, scale=scale)
            return rec_res

# ocr引擎初始化
ocr_engine = PaddleOCR()
lock = threading.Lock()

def main(img_path):
    try:
        lock.acquire() # 锁定
        # det:是否进行检测; rec: 是否进行识别; use_angle_cls: 是否角度分类
        result = ocr_engine.ocr(img_path, det=True, rec=True, cls=False)
        # 转化为思贤ocr格式
        blocks = ocr_format_convert(result,det=True, rec=True)
    finally:
        lock.release() # 释放
    return blocks

if __name__ == '__main__':
    img_path = r'C:\Users\admin\Desktop\lizi\1334018057747562496_tb.jpg'
    print(main(img_path))

