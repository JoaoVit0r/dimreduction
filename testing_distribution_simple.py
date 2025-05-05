targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
           11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
           21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
           41, 42, 43, 44, 45, 46, 47, 48, 49, 50,]
number_of_threads=10

threads = []
# Create groups with spaced items
groups = [[] for _ in range(number_of_threads)]
indices = [[] for _ in range(number_of_threads)]

# Distribute targets in a round-robin fashion
for i, target in enumerate(targets):
    thread_index = i % number_of_threads
    groups[thread_index].append(target)
    indices[thread_index].append(i)

# Start a thread for each group
for i, (group, offset_indices) in enumerate(zip(groups, indices)):
    print(f"Starting thread {i} with targets: {group}")
    print(f"Offset indices for thread {i}: {offset_indices}")