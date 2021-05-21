# -*- coding: utf-8 -*-
# @Time    : 2021/3/4
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : simple_funcs.py
# @Software: PyCharm
from configs.config import  h_ratio_threshold, lr_marg, A4_base_size, short_text
import numpy, copy, cv2

def ocr_format_convert(result,det=True, rec=False):
    blocks = []
    if result is not None:
        for line in result:
            temp = dict()
            text, score, chars_position, chars_score = '', 0.0, [], []
            if det and not rec:
                position = line
            elif det and rec:
                position, text_score, chars_position, chars_score = line
                text, score = text_score
            # 转为思贤ocr格式
            temp['characters'] = []
            temp['score'] = round(float(score),3)
            temp['text'] = text
            # temp['if_handwriting'] = 0
            temp['position'] = [{'x':int(point[0]),'y':int(point[1])}for point in position]
            # 字符信息
            for idx, chr_pos in enumerate(chars_position):
                ele = dict()
                ele['text'] = text[idx]
                ele['score'] = round(float(chars_score[idx]),3)
                ele['position'] = [{'x':int(point[0]), 'y':int(point[1])}for point in chr_pos]
                temp['characters'].append(ele)
            blocks.append(temp)
    return blocks

def point_in_rect(point, rect):
    flag = False
    x_min, y_min, x_max, y_max = rect['min_max']
    c1 = x_min<=point[0]<=x_max
    c2 = y_min<=point[1]<=y_max
    if c1 and c2: flag=True
    return flag

def delete_inner_rect(rects):
    rects_info = []
    for rect in rects:
        temp = dict()
        rect_np = numpy.array(rect)
        x_min,x_max = rect_np[:,0].min(),rect_np[:,0].max()
        y_min, y_max = rect_np[:, 1].min(), rect_np[:, 1].max()
        temp['min_max'], temp['rect'], temp['width'] = [x_min, y_min, x_max, y_max], rect, abs(x_max-x_min)
        rects_info.append(temp)
    rects_info = sorted(rects_info, key=lambda x: x['width']) # 宽度由小到大排序
    delete_rect = []
    for i, rect1 in enumerate(rects_info):
        for j, rect2 in enumerate(rects_info):
            if j<=i: continue
            temp = [point_in_rect(point, rect2) for point in rect1['rect']]
            if temp.count(True)==4:
                delete_rect.append(rect1['rect'])
    rects = [ele for ele in rects if ele not in delete_rect]
    return rects

def delete_inner_rect_old(rects):
    delete_rect = []
    for i, rect1 in enumerate(rects):
        for j, rect2 in enumerate(rects):
            if j==i: continue
            temp = [point_in_rect(point, rect2) for point in rect1]
            if temp.count(True)==4:
                delete_rect.append(rect1)
    for ele in delete_rect:
        if ele in rects:
            rects.remove(ele)
    return rects

def rect_next_rect(rect1, rect2):
    '''第一个版本'''
    flag, merge_rect = False, []
    x1_group, x2_group = [tmp[0] for tmp in rect1], [tmp[0] for tmp in rect2]
    y1_group, y2_group = [tmp[1] for tmp in rect1],[tmp[1] for tmp in rect2]
    x1_min, x1_max, x2_min, x2_max = min(x1_group), max(x1_group),min(x2_group), max(x2_group)
    y1_min, y1_max, y2_min, y2_max = min(y1_group), max(y1_group), min(y2_group), max(y2_group)
    c1 = x2_min<= x1_min<= x2_max
    c2 = x1_max-x2_max >= 0
    if c1 and c2:
        x_min, y_min = min(x1_min, x2_min), min(y1_min, y2_min)
        x_max, y_max = max(x1_max, x2_max), max(y1_max, y2_max)
        merge_rect = [[x_min,y_min],[x_max, y_min],[x_max, y_max],[x_min, y_max]]
        flag=True
    if not flag:
        # rect1在rect2左侧
        f1 = x2_min <= x1_max <= x2_min
        f2 = x2_max-x1_max >= 0
        if f1 and f2:
            x_min, y_min = min(x1_min, x2_min), min(y1_min, y2_min)
            x_max, y_max = max(x1_max, x2_max), max(y1_max, y2_max)
            merge_rect = [[x_min,y_min],[x_max, y_min],[x_max, y_max],[x_min, y_max]]
            flag=True
    return flag, merge_rect

def merge_next_rect(rects):
    delete_rects = []
    rect_f32 = [numpy.array(rect, dtype=numpy.float32) for rect in rects]
    minrect = [cv2.minAreaRect(rect) for rect in rect_f32]
    for i, rect1 in enumerate(rects):
        for j, rect2 in enumerate(rects):
            if j==i: continue
            # 高度是否相近
            h_sim = min(min(minrect[i][1])/min(minrect[j][1]), min(minrect[j][1])/min(minrect[i][1]))
            if h_sim < h_ratio_threshold: continue
            # 重心处于同一高度（对倾斜文本后续改进）
            if abs(minrect[i][0][1]-minrect[j][0][1]) > min([min(minrect[i][1]),min(minrect[j][1])])/2:
                continue
            # 水平方向是否相交
            flag, merge_rect = rect_next_rect(rect1, rect2)
            if flag:
                rects[i]=merge_rect
                if rect1 not in delete_rects: delete_rects.append(rect1)
                if rect2 not in delete_rects: delete_rects.append(rect2)
    for ele in delete_rects:
        if ele in rects:
            rects.remove(ele)
    return rects

def expand_lr_pixel(rects):
    for i, rect in enumerate(rects):
        for j, point in enumerate(rect):
            if j in [0,3]:
                point[0] = max(point[0] - lr_marg, 0)
            else:
                point[0] = point[0] + lr_marg # 可能越界？
    return rects

def  amend_short_text_box(result):
    for i, box in enumerate(result):
        x = [ele[0] for ele in box]
        y = [ele[1] for ele in box]
        if abs(min(x)-max(x))<short_text:
            result[i]=[[min(x),min(y)],[max(x),min(y)],[max(x),max(y)],[min(x),max(y)]]
    return result

def custom_postprocess(result):
    import time
    '''
    list进, list出; array 进 array 出
    :param result:
    :return:
    '''
    result_cp = copy.deepcopy(result)
    if type(result_cp) is numpy.ndarray: result=result.tolist()
    # 对于短文本，强制采用矩形框定位
    result = amend_short_text_box(result)
    # 后处理，如果A与B水平且紧邻，合并AB
    result = merge_next_rect(result)
    # 后处理,如果B中包含A，则删除A
    result = delete_inner_rect(result)
    # 后处理，拓展左右边界lr
    result = expand_lr_pixel(result)
    if type(result_cp) is numpy.ndarray: result = numpy.array(result)
    return result

def find_char_start_end_fr(label):
    '''
    chars_start_end = [{'label': 1779, 'start': 0, 'end': 3}, {'label': 1538, 'start': 3, 'end': 10}, {'label': 5209, 'start': 10, 'end': 19}, {'label': 15, 'start': 19, 'end': 24}, {'label': 149, 'start': 24, 'end': 31}, {'label': 5209, 'start': 31, 'end': 34}, {'label': 1087, 'start': 34, 'end': 38}, {'label': 32, 'start': 38, 'end': 41}]
    label_idx = [[1779, [3]], [1538, [10]], [5209, [19, 18, 17]], [15, [24]], [149, [31]], [5209, [34, 33]], [1087, [38]], [32, [41]]] # 衡量每个label占用哪几帧
    :param label:
    :return:
    '''
    frs_flag = [[True, ele, i] for i, ele in enumerate(label) if ele!=0]
    for i, fr in enumerate(frs_flag):
        flag, ele, idx = fr
        if i+1<len(frs_flag):
            if ele == frs_flag[i+1][1] and (idx+1==frs_flag[i+1][2]): frs_flag[i][0]=False
    end_frs = [ele for ele in frs_flag if ele[0] is True]
    chars_start_end = []
    for i, ele in enumerate(end_frs):
        start_end = dict()
        start_end['label'] = ele[1]
        if i==0: start_end['start']=0
        else:start_end['start'] = end_frs[i-1][-1]
        start_end['end'] = ele[-1]
        chars_start_end.append(start_end)
    for i, ele in enumerate(frs_flag):
        if ele[0] is False:
            frs_flag[i+1] = frs_flag[i+1] + frs_flag[i][2:]
    frs_flag = [ele for ele in frs_flag if ele[0] is True]
    label_idx = [[ele[1],ele[2:]] for ele in frs_flag]
    return chars_start_end, label_idx

def parse_char_pos_score(img_list, input_img_list, rec_result, rec_label, rec_prob, rec_rsz_pd):
    '''

    :param img_list: img list
    :param rec_result: (text, score) list
    :param rec_label: label list
    :param rec_prob: label每帧的置信度
    :param rec_rsz_pd: 每个rec的resize和padding信息(h,w,pd)
    :return: chars_info [[[(x1,y1),(x2,y2),(x3,y3),(x4,y4)],[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]],[]]
    '''
    chars_info = {'chars_position':[], 'block_hw':[], 'text':[], 'score':[]}
    imgs_hwc = [img.shape for img in img_list] # 原始图
    input_imgs_hwc = [(img.shape[1],img.shape[2],img.shape[0]) for img in input_img_list] # 输入到模型的图,包括paddling
    norm_img_h = input_imgs_hwc[0][0]
    imgs_frs = [len(label) for label in rec_label]
    imgs_frs_pixel = [w/imgs_frs[i] for i,(h,w,c) in enumerate(input_imgs_hwc)] # 约4pixel/帧
    for i, label in enumerate(rec_label):
        # chars_start_end用于求位置；label_idx用于求置信度
        # if rec_result[i][0]!= '登记机关': continue
        chars_start_end, label_idx = find_char_start_end_fr(label)
        # print('-------')
        # print('chars_start_end',len(chars_start_end), chars_start_end)
        # print(chars_start_end)
        # print('label', label)
        rsz_h, rsz_w, pd_pixel = rec_rsz_pd[i]
        # print('imgs_hwc', imgs_hwc[i], input_imgs_hwc[i], len(label), imgs_frs_pixel[i], rsz_w, pd_pixel)
        input_src_h_ratio = rsz_h / imgs_hwc[i][0]
        input_src_w_ratio = rsz_w / imgs_hwc[i][1]
        rec_points = []
        input_points = []
        rec_conf = []
        src_img_h = int(norm_img_h / input_src_h_ratio)
        for j, ele in enumerate(chars_start_end):
            start, end = ele['start'], ele['end']
            start_pixel = int(max(start*imgs_frs_pixel[i]-imgs_frs_pixel[i]/2,0))
            end_pixel = int(min(end*imgs_frs_pixel[i]-imgs_frs_pixel[i]/2,input_imgs_hwc[i][1]-pd_pixel))
            in_points = [(start_pixel, 0), (end_pixel, 0), (end_pixel, norm_img_h), (start_pixel, norm_img_h)]
            input_points.append(in_points) # input_points用于debug
            # 逆resize
            start_pixel = int(start_pixel/input_src_w_ratio)
            end_pixel = int(end_pixel/input_src_w_ratio)
            points = [(start_pixel,0),(end_pixel,0),(end_pixel, src_img_h), (start_pixel, src_img_h)]
            rec_points.append(points)
            # 计算字符cof
            frs_cof = [rec_prob[i][idx] for idx in label_idx[j][1]]
            rec_conf.append(round(sum(frs_cof)/len(frs_cof),3))
        # print('input_points', input_points)
        # print('rec_points', rec_points)
        assert len(rec_points)==len(chars_start_end) # 一一对应
        chars_info['chars_position'].append(rec_points)
        chars_info['block_hw'].append([imgs_hwc[i][0], imgs_hwc[i][1]])
        chars_info['text'].append(rec_result[i][0])
        chars_info['score'].append(rec_conf)
    # print('rec_result',rec_result)
    # print('imgs_hwc', imgs_hwc)
    # print('chars_info', chars_info)
    return chars_info

def amend_line_chars_pos(line_chars, line_box):
    if len(line_chars)==0: return line_chars
    r_border_x = line_box[1][0]
    left_margin_w = line_chars[0][1][0] - line_chars[0][0][0]
    # 其余字符位置整体往左移left_margin_w
    new_line_chars = []
    for points in line_chars:
         new_line_chars.append([(max(point[0] - left_margin_w, 0), point[1]) for point in points])
    # 附加最后一个box
    last_box = new_line_chars[-1]
    new_line_chars.append([last_box[1], (r_border_x, last_box[1][1]),(r_border_x, last_box[2][1]),last_box[2]])
    # 删除第一个box
    del new_line_chars[0] # 删除第一个四边形
    return new_line_chars

def inverse_perspective_rot(chars_info, perspective_rot_list):
    chars_position = chars_info['chars_position']
    block_hw = chars_info['block_hw']
    for i, chars_pos in enumerate(chars_position):
        h,w = block_hw[i]
        M, dist, rot_flag = perspective_rot_list[i]
        M_inv = numpy.linalg.inv(M)
        line_chars = []
        for j, points in enumerate(chars_pos):
            src_pts = [] # 四边形4个点
            for k, pt in enumerate(points):
                distant1 = dist[0] + (dist[1]-dist[0])*(pt[0]/w)
                distant2 = dist[3] + (dist[2] - dist[3]) * (pt[0] / w)
                distant = (distant1+distant2)/2
                pt = numpy.array([pt[0]*distant, pt[1]*distant, distant], dtype=numpy.float32)
                src_pt = numpy.dot(M_inv, pt) # 原图坐标
                src_pt = (round(src_pt[0]), round(src_pt[1]))
                src_pts.append(src_pt)
            line_chars.append(src_pts)
        chars_info['chars_position'][i] = line_chars
    return chars_info

def amend_chars_pos(chars_pos, dt_boxes):
    for i, line_chars in enumerate(chars_pos):
        line_chars = amend_line_chars_pos(line_chars, dt_boxes[i])
        chars_pos[i] = line_chars
    return chars_pos

def custom_img_resize_old(img):
    # A4_base_size = (800,1200) # (w, h)
    scale = 1.0
    height, width = img.shape[0:2]
    if width<=A4_base_size[0] and height<=A4_base_size[1]:
        return img, scale
    scale = max(width/A4_base_size[0], height/A4_base_size[1])
    img = cv2.resize(img, (int(width / scale), int(height / scale)))
    return img, scale

def custom_img_resize(img):
    scale = 1.0
    height, width = img.shape[0:2]
    if width<=A4_base_size and height<=A4_base_size:
        return img, scale
    scale = height / A4_base_size if height>=width else width/A4_base_size
    img = cv2.resize(img, (int(width / scale), int(height / scale)))
    return img, scale


def restore_img_size(dt_boxes=None, chars_box=None, scale=1.0):

    def restore_char_pos(chars_box):
        for i, sentence in enumerate(chars_box):
            for j, word in enumerate(sentence):
                for k, ele in enumerate(word):
                    chars_box[i][j][k] = (ele[0]*scale, ele[1]*scale)
        return chars_box

    if dt_boxes is not None and chars_box is not None:
        dt_boxes = [box * scale for box in dt_boxes]
        chars_box = restore_char_pos(chars_box)
        return dt_boxes, chars_box

    if dt_boxes is None and chars_box is not None:
        chars_box = restore_char_pos(chars_box)
        return chars_box

    if dt_boxes is not None and chars_box is None:
        dt_boxes = [box*scale for box in dt_boxes]
        return dt_boxes