# -*- coding: utf-8 -*-
# @Time    : 2021/3/4
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : config.py
# @Software: PyCharm

import argparse

# 部署
port = 9020
GPU = True
GPU_MEM = 3000
gpu_memory_optim = True # 如果限制GPU显存占用，需要设置为True
host = "0.0.0.0"
gpu_id = 0 # 0,1

# 训练参数
train_config_path = 'configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml'
# 自定义后处理逻辑
custom_handle = True # 是否执行自定义后处理逻辑
h_ratio_threshold = 0.7 # 待合并文本行的高度相近度
lr_marg = 2 # 左右边界扩展像素，取值范围[1,2,3,4]
# 如果A4为True,即输入为A4纸张，输入检测模型前进行相应的缩放；否,则原图输入检测模型
A4 = True # 以A4尺度比列缩放，否则采用原图
A4_base_size = 1888 # 1200
# minAreaRect文本框矩形还是四边形,如果True选择四边形
minAreaRect = True
# 短文本（10字以内）尺度(宽度)定义，对于短文本不使用四边形
short_text = 100

args = argparse.Namespace(
use_gpu = GPU,
ir_optim = True,
use_tensorrt = False,
gpu_mem = GPU_MEM,
image_dir = '',
det_algorithm = 'DB',
det_model_dir = './inference/det_db',
# det_model_dir = r'./ppocr/model_params/det',
det_max_side_len = 960,
det_limit_type='max',
det_db_thresh = 0.2, # 0.3(精确情况下)
det_db_box_thresh = 0.5,# 0.5
det_db_unclip_ratio = 1.6,# 1.6?
use_dilation = False,
det_east_score_thresh = 0.8,
det_east_cover_thresh = 0.1,
det_east_nms_thresh = 0.2,
rec_algorithm = 'CRNN',
# rec_model_dir = './ppocr/model_params/rec_server',
rec_model_dir = 'ppocr/model_params/rec_mobile',
# rec_model_dir = './ppocr/model_params/rec',
rec_image_shape = "3, 32, 320",
rec_char_type = 'ch',
rec_batch_num = 30,
max_text_length = 25,
rec_char_dict_path = './configs/ppocr_keys_v1.txt',
use_space_char = True,
drop_score = 0, # 0.5,剔除识别置信度低的文本
cls_model_dir = './ppocr/model_params/cls',
cls_image_shape = "3, 48, 192",
label_list = ['0', '180'],
cls_batch_num = 30,
cls_thresh = 0.9,
enable_mkldnn = False,
use_zero_copy_run = False,
use_pdserving = False,
lang = 'ch',
det = True,
rec = True,
use_angle_cls = True)




