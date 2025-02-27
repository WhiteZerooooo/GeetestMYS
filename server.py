from flask import Flask, request, jsonify
import os
import numpy as np
from crop_image import crop_image, convert_png_to_jpg
import time
from PIL import Image
from io import BytesIO
import requests
import onnxruntime as ort
import re
import json
import execjs
import random

app = Flask(__name__)

# 使用会话对象
sessionGet = requests.Session()
sessionGet.headers.update({
    "User-Agent": "",
    "cookie": "",
})

# 加载onnx模型
start = time.time()
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'resnet18.onnx')
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
print("加载模型，耗时:", time.time() - start)

badtime = 0

@app.errorhandler(Exception)
def handle_exception(e):
    # 返回错误码201和错误信息
    return jsonify({"error": str(e)}), 201

@app.route('/nine', methods=['GET'])
def nine():
    global badtime
    gt = request.args.get('gt')
    challenge = request.args.get('challenge')
    badtime = 0
    validate = Geetest(gt, challenge)

    # 构造响应数据
    res = {
        "gt": gt,
        "challenge": challenge,  # 示例值，可以根据需要调整
        "validate": validate  # 这里的 validate 可以动态生成或固定
    }
    
    response_data = {
        "msg": "",
        "data": res
    }
    
    return jsonify(response_data)

def process_image_from_url(image_url, save_bg_path='image_test/bg.jpg', save_icon_path='image_test/icon.png'):
    # 下载图片
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # 获取图像尺寸
    img_width, img_height = img.size
    upper_part = img.crop((0, 0, 344, 344))
    bottom_left_part = img.crop((0, 344, 40, 384))
    black_edge_width = 4
    tile_size = 344 // 3
    clean_tile_size = tile_size - 2 * black_edge_width

    # 新的无黑边九宫格图像大小
    new_image_size = clean_tile_size * 3  # 无黑边后的整体大小
    new_img = Image.new('RGB', (new_image_size, new_image_size))

    # 裁剪和去除黑边并拼接九宫格
    for i in range(3):  # 行
        for j in range(3):  # 列
            # 计算每块的左上角和右下角坐标，去除黑边
            left = j * tile_size + black_edge_width
            upper = i * tile_size + black_edge_width
            right = (j + 1) * tile_size - black_edge_width
            lower = (i + 1) * tile_size - black_edge_width
            tile = upper_part.crop((left, upper, right, lower))
            new_left = j * clean_tile_size
            new_upper = i * clean_tile_size
            new_img.paste(tile, (new_left, new_upper))

    new_img.save(save_bg_path)
    bottom_left_part.save(save_icon_path)

def predict_onnx(icon_image, bg_image):
    coordinates = [
        [1, 1],
        [1, 2],
        [1, 3],
        [2, 1],
        [2, 2],
        [2, 3],
        [3, 1],
        [3, 2],
        [3, 3],
    ]

    def cosine_similarity(vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity
    def data_transforms(image):
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = image_array.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_array = (image_array - mean) / std
        image_array = np.transpose(image_array, (2, 0, 1))
        # image_array = np.expand_dims(image_array, axis=0)
        return image_array
    target_images = []
    target_images.append(data_transforms(Image.open(BytesIO(icon_image))))
    bg_images = crop_image(bg_image, coordinates)
    for bg_image in bg_images:
        target_images.append(data_transforms(bg_image))
    start = time.time()
    outputs = session.run(None, {input_name: target_images})[0]
    scores = []
    for i, out_put in enumerate(outputs):
        if i == 0:
            target_output = out_put
        else:
            similarity = cosine_similarity(target_output, out_put)
            scores.append(similarity)
    print(scores)
    indexed_arr = list(enumerate(scores))
    index_to_grid = {
        0: "1_1", 1: "2_1", 2: "3_1", 3: "1_2", 4: "2_2", 
        5: "3_2", 6: "1_3", 7: "2_3", 8: "3_3"
    }
    filtered_arr = [item for item in indexed_arr if item[1] > 0.4]
    mapped_arr = [index_to_grid[item[0]] for item in filtered_arr]
    result = ','.join(mapped_arr)
    print(f"识别完成{result}，耗时: {time.time() - start}")
    return result

def Geetest(gt, challenge, c = [], s = "", isRe = False):
    data = {}
    c = c
    s = s
    if isRe:
        # 启动验证码1
        ts = int(time.time() * 1000)
        url = f"https://api.geevisit.com/refresh.php?gt={gt}&challenge={challenge}&lang=zh-cn&type=click&callback=geetest_{ts}"
        res = sessionGet.get(url)
        match = re.search(r'geetest_\d+\((.*)\)', res.text, re.DOTALL)
        json_str = match.group(1)
        data = json.loads(json_str)
        print(json.dumps(data, indent=4, ensure_ascii=False))
    else:
        # 启动验证码1
        ts = int(time.time() * 1000)
        url = f"https://api.geetest.com/get.php?gt={gt}&challenge={challenge}&lang=zh-cn&pt=0&client_type=web&callback=geetest_{ts}"
        res = sessionGet.get(url)
        match = re.search(r'geetest_\d+\((.*)\)', res.text, re.DOTALL)
        json_str = match.group(1)
        data = json.loads(json_str)
        print(json.dumps(data, indent=4, ensure_ascii=False))

        # 启动验证码2
        ts = int(time.time() * 1000)
        url = f"https://api.geevisit.com/ajax.php?gt={gt}&challenge={challenge}&lang=zh-cn&pt=0&client_type=web&callback=geetest_{ts}"
        res = sessionGet.get(url)
        match = re.search(r'geetest_\d+\((.*)\)', res.text, re.DOTALL)
        json_str = match.group(1)
        data = json.loads(json_str)
        print(json.dumps(data, indent=4, ensure_ascii=False))

        # 获取图片验证码
        ts = int(time.time() * 1000);
        url = f"https://api.geevisit.com/get.php?is_next=true&type=click&gt={gt}&challenge={challenge}&lang=zh-cn&https=false&protocol=https://&offline=false&product=embed&api_server=api.geetest.com&isPC=true&autoReset=true&width=100%25&callback=geetest_{ts}"
        res = sessionGet.get(url)
        match = re.search(r'geetest_\d+\((.*)\)', res.text, re.DOTALL)
        json_str = match.group(1)
        data = json.loads(json_str)
        print(json.dumps(data, indent=4, ensure_ascii=False))
        c = data["data"]['c']
        s = data["data"]['s']

    pic: str = 'https://static.geetest.com' + data["data"]['pic']

    # 请求验证码处理服务
    process_image_from_url(pic)
    with open("image_test/icon.png", "rb") as rb:
        icon_image = convert_png_to_jpg(rb.read())
    with open("image_test/bg.jpg", "rb") as rb:
        bg_image = rb.read()
    poses = predict_onnx(icon_image, bg_image)
    if poses == "":
        poses = "1_1"
    return do_verify(poses, pic, gt, challenge, c, s)


def do_verify(codes, pic, gt, challenge, c, s):
    global badtime
    stringCodes = codes
    print(stringCodes)
    print(
        f'处理后坐标: {stringCodes}',
        f'图片地址: {pic}',
        f'gt:{gt}, challenge:{challenge}',
        f'c: {c}, s: {s}', sep='\n',
    )

    # 执行 JavaScript 获取轨迹
    w = execjs.compile(open('main.js', 'r', encoding='utf-8').read()).call(
        'get_w',
        stringCodes, pic, gt, challenge, c, s
    )
    print(f'轨迹: {w}')

    # 避免出现点选过快的情况
    time.sleep(2)
    ts = int(time.time() * 1000);
    url = f"https://api.geevisit.com/ajax.php?gt={gt}&challenge={challenge}&lang=zh-cn&pt=0&client_type=web&w={w}&callback=geetest_{ts}"
    res = sessionGet.get(url)
    match = re.search(r'geetest_\d+\((.*)\)', res.text, re.DOTALL)
    json_str = match.group(1)
    data = json.loads(json_str)
    print("验证返回：\n", json.dumps(data, indent=4, ensure_ascii=False))

    time.sleep(random.randint(1, 3))
    if (data["data"]["result"] != "fail"):
        validate = data["data"]["validate"]
        return validate
    else:
        badtime = badtime + 1
        if badtime < 10:
            return Geetest(gt, challenge, c, s, True)
        else:
            return ""
    return ""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001)
