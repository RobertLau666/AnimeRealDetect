from concurrent.futures import ThreadPoolExecutor, as_completed
from inference import AnimeRealCls
import pandas as pd
import time
from tqdm import tqdm
from utils import get_current_time
import os


def process_single_input(classifier, input_data):
    try:
        start_time = time.time()
        anime_prob, real_prob, result = classifier(input_data)
        cost_time = time.time() - start_time
        return anime_prob, real_prob, result, cost_time
    except Exception as e:
        print(f"Error processing {input_data}: {str(e)}")
        return None, None, 'error', 0.0


if __name__ == "__main__":
    all_start_time = time.time()

    output_dir = 'data/my_test/output'
    os.makedirs(output_dir, exist_ok=True)

    # 初始化分类器
    classifier = AnimeRealCls(model_dir="model/caformer_s36_v1.3_fixed")

    # 读取输入 CSV
    csv_path = 'data/my_test/input/2025年5月12日-未成年标注 - 工作表1_style_label.csv'
    df = pd.read_csv(csv_path)
    test_inputs = df['image_path'].tolist()

    max_workers = 12  # 可根据机器调整

    # 并发执行分类任务
    url_to_result = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_input, classifier, url): url for url in test_inputs}
        for future in tqdm(as_completed(futures), total=len(futures)):
            url = futures[future]
            anime_prob, real_prob, result, cost_time = future.result()
            url_to_result[url] = (anime_prob, real_prob, result, cost_time)

    # 保证输出顺序与输入顺序一致
    anime_probs, real_probs, results, cost_times = [], [], [], []
    for url in test_inputs:
        anime_prob, real_prob, result, cost_time = url_to_result.get(url, (None, None, 'error', 0.0))
        anime_probs.append(anime_prob)
        real_probs.append(real_prob)
        results.append(result)
        cost_times.append(cost_time)

    # 写入结果
    df['anime_prob'] = anime_probs
    df['real_prob'] = real_probs
    df['result'] = results
    df['cost_time'] = cost_times

    output_filename = os.path.join(
        output_dir,
        f"{get_current_time()}_{os.path.splitext(os.path.basename(csv_path))[0]}_maxworkers_{max_workers}_result.csv"
    )
    df.to_csv(output_filename, index=False)
    print(f"Saved result to {output_filename}")

    all_end_time = time.time()
    print(f"inference_batch.py\nall_cost_time: {all_end_time - all_start_time} len(test_inputs): {len(test_inputs)} each_cost_time: {(all_end_time - all_start_time)/len(test_inputs)} max_workers: {max_workers}")
