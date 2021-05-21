# -*- coding: utf-8 -*-
# @Time    : 2021/10/3
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : test_http_server.py
# @Software: PyCharm
import  requests

def ocr(img_path):
    url = "http://10.0.96.22:9020/api/algorithm/extract_text_info"
    # url = "http://127.0.0.1:9020/api/algorithm/extract_text_info"
    files = {'file': ('test.jpg', open(img_path, 'rb'), 'image/jpeg')}
    timeout = 180
    r = requests.post(url, files=files, timeout=timeout)
    ocr_result = r.text
    return ocr_result

if __name__ == '__main__':
    path = r'C:\Users\admin\Desktop\lizi\1334020471129112576_tx_tb_sc.jpg'
    print(ocr(path))