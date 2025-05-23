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
    classifier = AnimeRealCls(model_dir="model/caformer_s36_v1.3_fixed")

    csv_path = '/data/code/chenyu.liu/others/AnimeRealDetect/data/my_test/input/测试1.csv'
    df = pd.read_csv(csv_path)
    if 'image_path' in df.columns:
        test_inputs = df['image_path'].tolist()
    elif 'image_url' in df.columns:
        test_inputs = df['image_url'].tolist()
    else:
        raise ValueError("Neither 'image_path' nor 'image_url' found in DataFrame columns.")

    max_workers = 30

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
