from datetime import datetime

def get_current_time():
    current_time = datetime.now()
    date_suffix = current_time.strftime("%Y%m%d") 
    time_suffix = current_time.strftime("%H%M%S") # 当然可以写为"_%H%M%S"，输出为：20240223_085639
    return date_suffix + time_suffix

if __name__ == "__main__":
    current_time = get_current_time()
    print(current_time)