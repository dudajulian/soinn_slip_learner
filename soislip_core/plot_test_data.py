import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from soinn_py import SoinnPlus

NUM_NOISE_POINTS =2000
NUM_SIGNAL_POINTS = 5000
NUM_SIGNAL2_POINTS = 5000
SCALE = 5
RING_SCALE = 0.5
SHUFFLE = False
DIMENSION = 2
LABELED_PERIOD = 100

# Create a simple 2D dataset (you can visualize this easily)
# Set random seed for reproducibility (optional)
np.random.seed(42)

# Generate random noise points
random_data = np.random.uniform(low=-SCALE, high=SCALE, size=(NUM_NOISE_POINTS, DIMENSION))
ring_labels = np.random.normal(loc=0.3, scale=0.1, size=NUM_SIGNAL_POINTS)
blob_labels = np.random.normal(loc=0.7, scale=0.1, size=NUM_SIGNAL2_POINTS)
random_labels = np.random.uniform(0, 1, NUM_NOISE_POINTS)


if DIMENSION == 2:
    # Generate ring points around center (2, 2) with radius between 0.7 and 1
    phi = np.random.uniform(0, 2 * np.pi, NUM_SIGNAL_POINTS)
    radii = np.random.uniform(0.8, 1.0, NUM_SIGNAL_POINTS)
    x_ring = radii * np.cos(phi)
    y_ring = radii * np.sin(phi)
    ring_data = np.column_stack((x_ring - 0.8, y_ring - 0.8))*SCALE*RING_SCALE
    blob_data = np.random.normal(loc=[2, 2], scale=0.5, size=(NUM_SIGNAL2_POINTS, 2))


if DIMENSION == 3:
    # Sample spherical coordinates
    phi = np.random.uniform(0, 2 * np.pi, NUM_SIGNAL_POINTS)           # Azimuthal angle (longitude)
    theta = np.arccos(np.random.uniform(-1, 1, NUM_SIGNAL_POINTS))     # Polar angle (latitude)
    radii = np.random.uniform(0.8, 1.0, NUM_SIGNAL_POINTS)
    # Convert to Cartesian coordinates
    x = radii * np.sin(theta) * np.cos(phi)
    y = radii * np.sin(theta) * np.sin(phi)
    z = radii * np.cos(theta)
    ring_data = np.column_stack((x, y, z)) * SCALE * RING_SCALE

# Label the datasets
ring_data = np.hstack((ring_data, ring_labels.reshape(-1, 1)))
blob_data = np.hstack((blob_data, blob_labels.reshape(-1, 1)))
random_data = np.hstack((random_data, random_labels.reshape(-1, 1)))

# Combine datasets
print(random_data.shape, ring_data.shape, blob_data.shape)
noisy_ring = np.vstack((ring_data, random_data[NUM_NOISE_POINTS//2:,:]))
noisy_blob = np.vstack((blob_data, random_data[:NUM_NOISE_POINTS//2,:]))
np.random.shuffle(noisy_ring)
np.random.shuffle(noisy_blob)
print(noisy_ring.shape, noisy_blob.shape)
data = np.vstack((noisy_ring, noisy_blob))
print(data.shape)

# Shuffle the combined dataset
if SHUFFLE:
    np.random.shuffle(data)

# Instantiate the SOINN+ object
soinn = SoinnPlus(dim=DIMENSION)

# Feed the signals one by one
for i, point in enumerate(data):
    soinn.input_signal(point[:-1]) 
    if i % LABELED_PERIOD == 0:
        soinn.input_signal(point[:-1], label=point[-1])  # Use the last column as label

# For now: just plot the nodes (weights)
nodes_nparray = np.vstack(soinn.nodes)
node_labels = np.array([p[0] if p[0] is not None else -0.1 for p in soinn.predictions])

if DIMENSION == 0:
    axis = [-SCALE, SCALE, -SCALE, SCALE]
    plt.scatter(random_data[:, 0], random_data[:, 1], color='black', label=f"Noise ({NUM_NOISE_POINTS} Samples)", s=1, alpha=0.5)
    plt.scatter(ring_data[:,0], ring_data[:,1], color='blue', label=f"Signal ({NUM_SIGNAL_POINTS} Samples)", s=1, alpha=0.5)
    plt.scatter(blob_data[:,0], blob_data[:,1], color='green', label=f"Signal2 ({NUM_SIGNAL2_POINTS} Samples)", s=1, alpha=0.5)
    plt.scatter(nodes_nparray[:,0], nodes_nparray[:,1], color='red', label=f"Nodes ({len(soinn.nodes)})", s=10, alpha=0.5)
    for j, neighbors in enumerate(soinn.adjacency_mat.rows):
        for k in neighbors:
            if k >= j:  # avoid duplicate lines
                nj = nodes_nparray[j]
                nk = nodes_nparray[k]
                plt.plot([nj[0], nk[0]], [nj[1], nk[1]], 'r')
    plt.legend()
    plt.axis(axis)
    if SHUFFLE:
        plt.title("Input (shuffled)")
    else:
        plt.title("Input (sorted)")


if DIMENSION == 2:
    print(nodes_nparray.shape, node_labels.shape)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(random_data[:,0], random_data[:,1], random_data[:,2], c='black', s=1, alpha=0.5, label=f"Noise ({NUM_NOISE_POINTS} Samples)")
    ax.scatter(ring_data[:,0], ring_data[:,1], ring_data[:,2], c='blue', s=5, alpha=0.5, label=f"Signal ({NUM_SIGNAL_POINTS} Samples)")
    ax.scatter(blob_data[:,0], blob_data[:,1], blob_data[:,2], c='green', s=5, alpha=0.5, label=f"Signal2 ({NUM_SIGNAL2_POINTS} Samples)")
    ax.scatter(nodes_nparray[:,0], nodes_nparray[:,1], node_labels, c='r', s=20, alpha=0.9, label=f"Nodes ({len(soinn.nodes)})")

    for j, neighbors in enumerate(soinn.adjacency_mat.rows):
        for k in neighbors:
            if k >= j:  # avoid duplicate lines
                nj = nodes_nparray[j]
                nk = nodes_nparray[k]
                ax.plot([nj[0], nk[0]], [nj[1], nk[1]], [node_labels[j], node_labels[k]], 'r')

    # Optional: set aspect ratio and labels
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-SCALE*RING_SCALE, SCALE*RING_SCALE])
    ax.set_ylim([-SCALE*RING_SCALE, SCALE*RING_SCALE])
    ax.set_zlim([-0.1, 1])
    ax.legend()

    if SHUFFLE:
        ax.set_title("Input (shuffled)")
    else:
        ax.set_title("Input (sorted)")

plt.show()


# soinn.show(save=False, save_path="soinn_output.png")