import matplotlib.pyplot as plt


# Data
data1 = [(1000, 0.07), (3000, 1.25), (5000, 5.36), 
         (7000, 14.36), (9000, 32.28), (11000, 55.28), 
         (13000, 92.36), (15000, 138.66)]


# Extract the data
N_values = [d[0] for d in data1]
runtime_values = [d[1] for d in data1]


# Plot
plt.figure(figsize=(10, 6))
plt.plot(N_values, runtime_values, marker='o', linestyle='-', color='b', label='Series 1')
plt.xlabel('Matrix Size (N)', fontsize=14)
plt.ylabel('Runtime (seconds)', fontsize=14)
plt.title('for Block Size = 100, Threads = 4', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)


# Adjust x-ticks to prevent overlap
plt.xticks(N_values, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()



# 2. Plotting Runtime vs. Block Size


# Data for different block sizes
block_sizes = [100, 1000, 10, 300, 150]
runtimes = [32.28, 247.26, 242.45, 49.61, 36.84]


# Sort data based on block size
sorted_data = sorted(zip(block_sizes, runtimes))
sorted_block_sizes, sorted_runtimes = zip(*sorted_data)


# Plotting
plt.figure(figsize=(8, 6))
plt.plot(sorted_block_sizes, sorted_runtimes, marker='o', linestyle='-', color='g')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Block Size', fontsize=14)
plt.ylabel('Runtime (seconds)', fontsize=14)
plt.title('for Threads = 4, N = 9000)', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(sorted_block_sizes)
plt.tight_layout()
plt.show()


# 3. Plotting Runtime vs. Number of Threads


# Data
threads = [1, 2, 3, 4, 5, 6, 7, 8]
runtime = [109.49, 59.22, 43.63, 32.28, 28.43, 26.49, 24.84, 24.83]


# Plot
plt.figure(figsize=(8, 6))
plt.plot(threads, runtime, marker='o', linestyle='-', color='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Threads', fontsize=14)
plt.ylabel('Runtime (seconds)', fontsize=14)
plt.title('Runtime vs. Number of Threads', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(threads)
plt.tight_layout()
plt.show()


