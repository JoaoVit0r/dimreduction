import threading
import random
import time
from main_from_cli import IOFile, VERBOSE_LEVEL 
IOFile.print_and_log(f"--------------------------------", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])

targets = [f"{i}" for i in range(100)]
txt = ["\n--------------------------------\n"]

# Process a single target and return results
def process_target(target):
    # Add a short random sleep to better demonstrate concurrency
    sleep_time = random.uniform(0.1, 0.5)
    time.sleep(sleep_time)
    
    targetindex = int(target)
    thread_id = threading.get_ident()
    IOFile.print_and_log(f"[THREAD {thread_id}] Target {target} PROCESSING - generating training set", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
    
    return {
        "targetindex": targetindex,
        "sleep_time": sleep_time
    }

results = [None] * len(targets)  # Preallocate the results list with None values
group_size = 10

def process_target_wrapper(target, index):
    thread_id = threading.get_ident()
    start_time = time.time()
    IOFile.print_and_log(f"[THREAD {thread_id}] Target {target} STARTED at {start_time}", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
    result = process_target(target)
    end_time = time.time()
    IOFile.print_and_log(f"[THREAD {thread_id}] Target {target} ENDED at {end_time} (Duration: {end_time - start_time:.4f}s)", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
    
    results[index] = {
        "index": index,
        "targetindex": result["targetindex"],
        "start_time": start_time,
        "end_time": end_time,
        "duration": end_time - start_time,
        "sleep_time": result["sleep_time"]
    }

for i in range(0, len(targets), group_size):
    group = targets[i:i + group_size]
    threads = []
    
    IOFile.print_and_log(f"Starting group {i//group_size + 1} with targets: {group}", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
    
    for j, target in enumerate(group):
        index = i + j
        thread = threading.Thread(target=process_target_wrapper, args=(target, index))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    IOFile.print_and_log(f"Completed group {i//group_size + 1}", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])

print("writing results sequentially")
with open("timing/results.log", "a") as f:
    f.write("index target start_time end_time duration sleep_time\n")
for result in results:
    targetindex = result["targetindex"]
    # txt.extend(result["txt"])
    
    # # save result in a file
    # with open("timing/txt.log", "a") as f:
    #     f.write(result["txt"])
    with open("timing/results.log", "a") as f:
        f.write(f"{result['index']} {result['targetindex']} {result['start_time']} {result['end_time']} {result['duration']} {result['sleep_time']}\n")
        
# print(txt)