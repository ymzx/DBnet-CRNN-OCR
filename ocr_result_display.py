# -*- coding: utf-8 -*-
import requests, json
requests.packages.urllib3.disable_warnings()
import cv2, os, numpy
from PIL import Image, ImageDraw, ImageFont
from paddleocr import main
import numpy as np

def cv2ImgAddText(img, text, left, top, textColor=(0, 0, 255), textSize=40):
    if (isinstance(img, numpy.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top-40), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)


def ocr_api(file_address, url):
    payload = open(file_address, 'rb')
    headers = {'Content-Type': 'image/jpeg'}
    r = requests.request("POST", url, headers=headers, data=payload, verify=False)
    result = json.loads(r.text)
    return result


def draw_rect(img_path, show_char=False):
    result = {'error_msg': 'ok', "status_code": 200, "blocks": [], "time_cost": str(0) + 's', 'angle': 0}
    result['blocks'] = main(img_path)
    print('ocr result', result)
    image = cv2.imread(img_path)
    for block in result['blocks']:
        position = block['position']
        rec = [[int(ele['x']), int(ele['y'])] for ele in position]
        points = np.array(rec)
        # xmin, xmax = position[0]['x'], position[2]['x']
        # ymin, ymax = position[0]['y'], position[2]['y']
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        # cv2.polylines(image, [points], True, (255, 0, 0), 2)
        if show_char:
            for chr_info in block['characters']:
                chr_position = chr_info['position']
                rec = [[int(ele['x']), int(ele['y'])] for ele in chr_position]
                points = np.array(rec)
                cv2.polylines(image, [points], True, (255, 0, 0), 2)
        else:
            cv2.polylines(image, [points], True, (255, 0, 0), 2)
        # image = cv2ImgAddText(image, str(block['text']), int(xmin), int(ymin))
        # print(position[0], position[2], "||", round(block['score'],3))
    _ , filename = os.path.split(img_path)
    output_path = os.path.join(r"C:\Users\admin\Desktop\ocr_test\debug_det", filename)
    cv2.imwrite(output_path, image)
    print("已保存至：", output_path)


if __name__ == "__main__":
    # imgs_path = r'C:\Users\admin\Desktop\ocr_test\batch_imgs'
    # imgs_dirs = os.listdir(imgs_path)
    # for ele in imgs_dirs:
    #     print(ele)
    #     img_path = os.path.join(imgs_path, ele)
    #     draw_rect(img_path, show_char=False)
    img_path = r'C:\Users\admin\Desktop\temp_jpg\1334012418585526272.jpg'
    # img_path = r'C:\Users\admin\Desktop\lizi\1336569206329573376.jpg'
    draw_rect(img_path, show_char=False)

