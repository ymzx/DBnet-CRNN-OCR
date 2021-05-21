# -*- coding: utf-8 -*-
# @Time    : 2021/2/21
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : ocr_api.py
# @Software: PyCharm
from flask_restplus import Resource, Api, reqparse
from flask import make_response, jsonify
from werkzeug.datastructures import FileStorage
from paddleocr import main
from ppocr.utils.log import get_logger
import numpy as np, cv2, time, uuid, traceback

logger = get_logger()

api = Api(version='2.0.0', title='算法API', description='A4文档字符识别',default='Paddle-based OCR',default_label='v2.0')
parser = reqparse.RequestParser()
parser.add_argument('file', type=FileStorage, location='files', required=True, help='Select ID to be recognition')

@api.route('/api/algorithm/extract_text_info')
@api.expect(parser)

class TableCutInfoExtract(Resource):
    @api.response(200, 'ok')
    @api.response(300, 'model server error')
    @api.response(400, 'request error')

    def post(self):
        result = {'error_msg': 'ok', "status_code": 200, "blocks": [], "time_cost": str(0) + 's', 'angle': 0}
        #-----获取传参 TCP耗时-------
        args = parser.parse_args()
        img = args['file']
        img_bytes = img.read()
        nparr = np.fromstring(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # 转为opencv格式,cv2.COLOR_RGB2BGR
        #----保存图片----
        t1 = time.time()
        src_image_path = str(uuid.uuid4().hex)+'.jpg'
        try:
            # -----OCR-------#
            blocks = main(image)
            result['blocks'] = blocks
        except Exception as error:
            cv2.imwrite(src_image_path, image)
            logger.error('api Exception:  ' + repr(error))
            logger.error('Api Exception Detail:  ' + str(traceback.format_exc()))
            result['error_msg'] = str(error)
            result['status_code'] = 300
        t2 = time.time()
        result['time_cost'] = str(round(t2-t1,3))+'s'
        return make_response(jsonify(result))

    def get(self):
        return {'hello': 'world'}