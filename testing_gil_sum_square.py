import threading

def sum_of_squares(n):
    return sum(i * i for i in range(n))

def worker():
    result = sum_of_squares(10**8)
    print(f'Result: {result}')

threads = []
for _ in range(4):
    t = threading.Thread(target=worker)
    threads.append(t)
    t.start()

for t in threads:
    t.join()