import gradio as gr
import pandas as pd
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import chardet
import tempfile
from inference import AnimeRealCls
from utils import get_current_time

def detect_encoding(file_path):
    """更健壮的编码检测"""
    try:
        with open(file_path, 'rb') as f:
            rawdata = f.read(100000)  # 读取更多数据提高检测准确性
            result = chardet.detect(rawdata)
        return result['encoding'] or 'utf-8'
    except Exception as e:
        print(f"编码检测失败，默认使用utf-8: {str(e)}")
        return 'utf-8'

def process_single_input(classifier, input_data):
    try:
        start_time = time.time()
        anime_prob, real_prob, result = classifier(input_data)
        cost_time = time.time() - start_time
        return anime_prob, real_prob, result, cost_time
    except Exception as e:
        print(f"Error processing {input_data}: {str(e)}")
        return None, None, 'error', 0.0

def run_inference(csv_file):
    all_start_time = time.time()

    # 创建临时文件副本处理Gradio文件对象
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
        # 将上传的文件内容写入临时文件
        with open(csv_file.name, 'rb') as f:
            tmp_file.write(f.read())
        tmp_path = tmp_file.name

    try:
        # 检测编码并读取CSV
        encoding = detect_encoding(tmp_path)
        print(f"检测到的编码: {encoding}")
        
        # 尝试多种编码方案
        encodings_to_try = [encoding, 'utf-8', 'gbk', 'gb18030', 'utf-16', 'latin1']
        df = None
        
        for enc in encodings_to_try:
            try:
                df = pd.read_csv(tmp_path, encoding=enc)
                print(f"成功使用编码: {enc}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise ValueError("无法用任何编码方案读取CSV文件")
            
        # 验证必须的列是否存在
        if 'image_path' not in df.columns:
            raise ValueError("CSV文件中缺少 'image_path' 列")
            
        test_inputs = df['image_path'].tolist()

        output_dir = 'data/my_test/output'
        os.makedirs(output_dir, exist_ok=True)

        classifier = AnimeRealCls(model_dir="model/caformer_s36_v1.3_fixed")

        max_workers = 12
        url_to_result = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_input, classifier, url): url for url in test_inputs}
            for future in tqdm(as_completed(futures), total=len(futures)):
                url = futures[future]
                anime_prob, real_prob, result, cost_time = future.result()
                url_to_result[url] = (anime_prob, real_prob, result, cost_time)

        anime_probs, real_probs, results, cost_times = [], [], [], []
        for url in test_inputs:
            anime_prob, real_prob, result, cost_time = url_to_result.get(url, (None, None, 'error', 0.0))
            anime_probs.append(anime_prob)
            real_probs.append(real_prob)
            results.append(result)
            cost_times.append(cost_time)

        df['anime_prob'] = anime_probs
        df['real_prob'] = real_probs
        df['result'] = results
        df['cost_time'] = cost_times

        output_filename = os.path.join(
            output_dir,
            f"{get_current_time()}_result.csv"
        )
        df.to_csv(output_filename, index=False, encoding='utf-8-sig')  # 使用utf-8-sig确保兼容性

        all_end_time = time.time()
        print(f"总耗时: {all_end_time - all_start_time:.2f}s, 平均每张: {(all_end_time - all_start_time)/len(test_inputs):.2f}s")

        return output_filename
        
    finally:
        # 确保临时文件被删除
        try:
            os.unlink(tmp_path)
        except:
            pass

def gradio_interface(csv_file):
    try:
        output_file = run_inference(csv_file)
        return output_file
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        raise gr.Error(f"处理失败: {str(e)}")

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.File(label="上传 CSV 文件（包含 image_path 列）"),
    outputs=gr.File(label="下载结果 CSV"),
    title="Anime vs Real 分类器（批量推理）",
    description="上传一个包含 image_path 列的 CSV 文件，点击按钮开始推理，完成后可下载结果文件。",
)

if __name__ == "__main__":
    iface.launch(server_port=7879, share=True)