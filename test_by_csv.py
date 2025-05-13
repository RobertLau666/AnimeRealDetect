import json
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import Union
from imgutils.data import load_image, rgb_encode
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
import pandas as pd
import time
from tqdm import tqdm
from utils import get_current_time
import os


class AnimeRealCls():
    def __init__(self, model_dir: str):
        self.session = requests.Session()  # Reuse session for HTTP requests
        self.model = self.load_local_onnx_model(f'{model_dir}/model.onnx')
        with open(f'{model_dir}/meta.json', 'r') as f:
            self.labels = json.load(f)['labels']
            
    def load_local_onnx_model(self, model_path: str) -> InferenceSession:
        """加载ONNX模型"""
        options = SessionOptions()
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            return InferenceSession(model_path, options)
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {str(e)}")

    # def _load_image(self, image_input: Union[str, bytes]) -> Image.Image:
    #     """加载图片，支持本地路径和HTTP URL"""
    #     try:
    #         if isinstance(image_input, bytes):
    #             # 如果是字节流
    #             return Image.open(BytesIO(image_input))
    #         elif image_input.startswith(('http://', 'https://')):
    #             # 如果是HTTP URL
    #             response = requests.get(image_input, timeout=10)
    #             response.raise_for_status()
    #             return Image.open(BytesIO(response.content))
    #         else:
    #             # 本地文件路径
    #             return Image.open(image_input)
    #     except Exception as e:
    #         raise ValueError(f"Failed to load image: {str(e)}")

    def _load_image(self, image_input: Union[str, bytes]) -> Image.Image:
        """加载图片，支持本地路径、HTTP URL 和字节流，优化速度"""
        try:
            if isinstance(image_input, bytes):
                # 字节流，直接读取
                return Image.open(BytesIO(image_input)).convert('RGB')
            elif isinstance(image_input, str) and image_input.startswith(('http://', 'https://')):
                # HTTP 请求优化
                with self.session.get(image_input, stream=True, timeout=5) as response:
                    response.raise_for_status()
                    return Image.open(response.raw).convert('RGB')  # 直接用 raw 提高效率
            elif isinstance(image_input, str) and os.path.exists(image_input):
                # 本地路径直接读取
                return Image.open(image_input).convert('RGB')
            else:
                raise ValueError("Unsupported image input format.")
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

    def _img_encode(self, image_input: Union[str, bytes], size=(384, 384), normalize=(0.5, 0.5)) -> np.ndarray:
        """图片编码预处理"""
        try:
            t1 = time.time()
            image = self._load_image(image_input) # 2.6596295833587646
            t2 = time.time()
            print("t2-t1: ", t2-t1)
            image = load_image(image, mode='RGB')
            t3 = time.time()
            print("t3-t2: ", t3-t2)
            image = image.resize(size, Image.BILINEAR)
            data = rgb_encode(image, order_='CHW')
            
            if normalize:
                mean_, std_ = normalize
                mean = np.asarray([mean_]).reshape((-1, 1, 1))
                std = np.asarray([std_]).reshape((-1, 1, 1))
                data = (data - mean) / std
            t4 = time.time()
            print("t4-t3: ", t4-t3)
            return data.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")

    def __call__(self, image_input: Union[str, bytes]) -> str:
        """执行分类"""
        try:
            t5 = time.time()
            input_ = self._img_encode(image_input, size=(384, 384))[None, ...]
            t6 = time.time()
            print("t6-t5: ", t6-t5)
            output, = self.model.run(['output'], {'input': input_}) # 0.09858536720275879
            t7 = time.time()
            print("t7-t6: ", t7-t6)
            values = dict(zip(self.labels, map(lambda x: x.item(), output[0])))
            anime_prob, real_prob = values['anime'], values['real']
            result = max(values, key=values.get)
            return anime_prob, real_prob, result
        except Exception as e:
            raise RuntimeError(f"Classification failed: {str(e)}")


if __name__ == "__main__":
    output_dir = 'data/my_test/output'
    classifier = AnimeRealCls(model_dir="model/caformer_s36_v1.3_fixed")
    
    # 测试数据（包含本地路径和HTTP URL）
    # test_inputs = [
    #     "1.webp",
    #     "2.webp",
    #     "3.webp",
    #     "https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc/729183_2110_28238288_1739099486934293155.webp?x-oss-process=image/resize,w_1080/format,webp",
    #     "https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc/729183_2113_28237810_1739013948135004916.webp?x-oss-process=image/resize,w_1080/format,webp",
    #     "https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc/729183_2113_28236856_1739010283114475558.webp?x-oss-process=image/resize,w_1080/format,webp",
    #     "https://ali-us-sync-image.oss-us-east-1.aliyuncs.com/linky_imggen_ugc/729183_2113_28238074_1739014906353175139.webp?x-oss-process=image/resize,w_1080/format,webp",
    # ]

    csv_path = 'data/my_test/input/ab实验真人-23个角色-character_id.csv'
    df = pd.read_csv(csv_path)
    test_inputs = df['img_url'].tolist()

    anime_probs, real_probs, results, cost_times = [], [], [], []
    for input_data in tqdm(test_inputs):
        try:
            start_time = time.time()
            anime_prob, real_prob, result = classifier(input_data)
            cost_time = time.time() - start_time
            anime_probs.append(anime_prob)
            real_probs.append(real_prob)
            results.append(result)
            cost_times.append(cost_time)
        except Exception as e:
            print(f"Error: {str(e)}")

    df[['anime_prob', 'real_prob', 'result', 'cost_time']] = list(zip(anime_probs, real_probs, results, cost_times))
    output_filename = os.path.join(output_dir, f"{get_current_time()}_{os.path.splitext(os.path.basename(csv_path))[0]}_result.csv")
    df.to_csv(output_filename, index=False)
    print(f"Saved result to {output_filename}")