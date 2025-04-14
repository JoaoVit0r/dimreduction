import threading
import random
import time
import json
from main_from_cli import IOFile, VERBOSE_LEVEL 

IOFile.print_and_log(f"--------------------------------", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])

# Load target distribution from file
try:
    with open("timing/target_distribution.json", "r") as f:
        target_heavy_levels = json.load(f)
except FileNotFoundError:
    print("Error: target_distribution.json not found. Please run generate_target_distribution.py first.")
    exit(1)

# Get all targets from the distribution
targets = list(target_heavy_levels.keys())[:60]
txt = ["\n--------------------------------\n"]

def cpu_heavy(level):
    sum(i * i for i in range(10**int(level)))
    return level

# Process a single target and return results
def process_target(target):
    targetindex = int(target)
    thread_id = threading.get_ident()
    IOFile.print_and_log(f"[THREAD {thread_id}] Target {target} PROCESSING - generating training set", path="timing/thread_execution.log", verbosity=VERBOSE_LEVEL["TIMER"])
    
    heavy_level = target_heavy_levels[target]
    cpu_heavy(heavy_level)
    
    return {
        "targetindex": targetindex,
        "heavy_level": heavy_level
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
        "heavy_level": result["heavy_level"]
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
    f.write("index target start_time end_time duration heavy_level\n")
for result in results:
    with open("timing/results.log", "a") as f:
        f.write(f"{result['index']} {result['targetindex']} {result['start_time']} {result['end_time']} {result['duration']} {result['heavy_level']}\n")