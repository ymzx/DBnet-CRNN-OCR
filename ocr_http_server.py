# -*- coding: utf-8 -*-
# @Time    : 2021/2/21
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : ocr_http_server.py
# @Software: PyCharm
from flask import Flask
from ocr_api import api
from configs.config import port, host

app = Flask(__name__) #  Create a Flask WSGI application
api.init_app(app)

app.run(host=host, port=port, debug=False)