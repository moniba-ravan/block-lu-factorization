import matplotlib.pyplot as plt

# Data
data = [
    (1000, 0.07),
    (3000, 1.25),
    (5000, 5.36),
    (7000, 14.36),
    (9000, 32.28),
    (11000, 55.28),
    (13000, 92.36),
    (15000, 138.664),
    (1000, 0.07),
    (3000, 1.25),
    (5000, 5.36),
    (7000, 14.36),
    (9000, 32.28),
    (11000, 55.28),
    (13000, 92.36),
    (15000, 138.66)
]

# Separate data for two series
data1 = [(n, runtime) for n, runtime in data[:8]]

# Plot
plt.figure(figsize=(10, 6))
plt.plot([d[0] for d in data1], [d[1] for d in data1], label='Series 1')
plt.xlabel('N')
plt.ylabel('Runtime (seconds)')
# plt.title('Runtime')
# plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------------

# Data for different block sizes
block_sizes = [100, 1000, 10, 300, 150]
runtimes = [32.28, 247.26, 242.45, 49.61, 36.84]

# Combine block sizes and runtimes into pairs
data = list(zip(block_sizes, runtimes))

# Sort data based on block size
data.sort(key=lambda x: x[0])

# Unzip sorted data
sorted_block_sizes, sorted_runtimes = zip(*data)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(sorted_block_sizes, sorted_runtimes, marker='o', linestyle='-')
# plt.title('Runtime vs. Block Size (4 threads, N=9000)')
plt.xlabel('Block Size')
plt.ylabel('Runtime (seconds)')
plt.grid(True)
# for x, y in data:
#     plt.text(x, y, f'{y:.2f}', ha='right', va='bottom')
plt.show()

# -------------------------------------------------------------------------
import matplotlib.pyplot as plt

# Data
threads = [1, 2, 3, 4, 5, 6, 7, 8]
runtime = [109.49, 59.22, 43.63, 32.28, 28.43, 26.49, 24.84, 24.83]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(threads, runtime, marker='o', linestyle='-')
# plt.title('Runtime vs Number of Threads')
plt.xlabel('Number of Threads')
plt.ylabel('Runtime (seconds)')
plt.grid(True)
plt.xticks(threads)

# for x, y in zip(threads, runtime):
#     plt.text(x, y, f'{y:.2f}', ha='right', va='bottom')


plt.show()
