#!/usr/bin/env python3

import numpy as np
import json
import os
import matplotlib.pyplot as plt
# Generate 5000 targets
num_targets = 5000
targets = [f"{i}" for i in range(num_targets)]

# Generate heavy levels using normal distribution
size=5000
mean=7.5
std_dev=1.0
data = np.random.seed(42)
data = np.random.normal(loc=mean, scale=std_dev, size=size)
rounded = np.round(data).astype(int)
clipped = np.clip(rounded, 6, 9)
heavy_levels = clipped
plt.hist(heavy_levels)
plt.show()

# Create target to heavy level mapping
target_distribution = {target: int(level) for target, level in zip(targets, heavy_levels)}

# Create output directory if it doesn't exist
os.makedirs("timing", exist_ok=True)

# Save distribution to JSON file
with open("timing/target_distribution.json", "w") as f:
    json.dump(target_distribution, f, indent=4)

# Save distribution to CSV for easy viewing
with open("timing/target_distribution.csv", "w") as f:
    f.write("target,heavy_level\n")
    for target, level in target_distribution.items():
        f.write(f"{target},{level:.4f}\n")

print(f"Generated distribution for {num_targets} targets")
print(f"Mean heavy level: {np.mean(heavy_levels):.4f}")
print(f"Standard deviation: {np.std(heavy_levels):.4f}")
print(f"Min heavy level: {np.min(heavy_levels):.4f}")
print(f"Max heavy level: {np.max(heavy_levels):.4f}")
print("\nDistribution saved to:")
print("- timing/target_distribution.json")
print("- timing/target_distribution.csv") 