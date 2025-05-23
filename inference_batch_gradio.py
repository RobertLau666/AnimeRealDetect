import gradio as gr
import pandas as pd
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from inference import AnimeRealCls
from utils import get_current_time
import chardet
import argparse

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))
    return result['encoding'] or 'utf-8'

def process_single_input(classifier, input_data):
    try:
        start_time = time.time()
        anime_prob, real_prob, result = classifier(input_data)
        cost_time = time.time() - start_time
        return anime_prob, real_prob, result, cost_time
    except Exception as e:
        print(f"Error processing {input_data}: {str(e)}")
        return None, None, 'error', 0.0

def run_inference(uploaded_file):
    original_filename = os.path.basename(uploaded_file.name)
    all_start_time = time.time()

    # 判断文件扩展名
    ext = os.path.splitext(original_filename)[-1].lower()

    if ext == ".csv":
        encoding = detect_encoding(uploaded_file.name)
        df = pd.read_csv(uploaded_file.name, encoding=encoding)
    elif ext == ".xlsx":
        df = pd.read_excel(uploaded_file.name)
    else:
        raise ValueError("只支持 .csv 或 .xlsx 格式的文件。")

    if 'image_path' in df.columns:
        test_inputs = df['image_path'].tolist()
    elif 'image_url' in df.columns:
        test_inputs = df['image_url'].tolist()
    else:
        raise ValueError("文件中缺少 image_path 或 image_url 列。")

    output_dir = 'data/my_test/output'
    os.makedirs(output_dir, exist_ok=True)

    classifier = AnimeRealCls(model_dir="model/caformer_s36_v1.3_fixed")

    max_workers = 30
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

    # 保存为 xlsx 文件（无论上传的是 csv 还是 xlsx）
    output_filename = os.path.join(
        output_dir,
        f"{get_current_time()}_{original_filename.replace('.csv', '').replace('.xlsx', '')}_result.xlsx"
    )
    df.to_excel(output_filename, index=False)

    all_end_time = time.time()
    print(f"all_cost_time: {all_end_time - all_start_time:.2f}s, each_cost_time: {(all_end_time - all_start_time)/len(test_inputs):.2f}s")

    return output_filename

def gradio_interface(uploaded_file):
    return run_inference(uploaded_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_port", type=int, default=7881, help="Port to launch the server on")
    args = parser.parse_args()

    iface = gr.Interface(
        fn=gradio_interface,
        inputs=gr.File(label="上传 CSV 或 Excel 文件（包含 image_path 或 image_url 列）", file_types=[".csv", ".xlsx"]),
        outputs=gr.File(label="下载结果 Excel 文件"),
        title="Anime vs Real 分类器（批量推理）",
        description="上传一个包含 image_path 或 image_url 列的 .csv 或 .xlsx 文件，点击按钮开始推理，完成后可下载结果文件（统一保存为 .xlsx）。",
    )
    iface.launch(server_port=args.server_port, share=True)
