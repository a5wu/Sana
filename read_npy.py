import numpy as np

# Load the .npy file
file_path = 'data/toy_data/00000000/0_1.npy'
data = np.load(file_path)

# Print basic information
print(f"Data shape: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Min value: {data.min()}")
print(f"Max value: {data.max()}")

# If it's not too large, print a small sample
if data.size < 50:
    print(f"Full data: {data}")
else:
    if len(data.shape) == 1:
        print(f"First 10 elements: {data[:10]}")
    elif len(data.shape) == 2:
        print(f"Top-left corner (5x5):\n{data[:5, :5]}")
    elif len(data.shape) == 3:
        print(f"Sample slice (first channel, 5x5):\n{data[0, :5, :5]}")
    elif len(data.shape) == 4:
        print(f"Sample slice (first item, first channel, 5x5):\n{data[0, 0, :5, :5]}") 