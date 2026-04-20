import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from soinn_py import SoinnPlus

NUM_NOISE_POINTS = 100
NUM_SIGNAL_POINTS = 500
SCALE = 10
RING_SCALE = 0.5
SHUFFLE = True
DIMENSION = 2

# Create a simple 2D dataset (you can visualize this easily)
# Set random seed for reproducibility (optional)
np.random.seed(42)

# Generate random noise points
random_data = np.random.uniform(low=-SCALE, high=SCALE, size=(NUM_NOISE_POINTS, DIMENSION))


if DIMENSION == 2:
    # Generate ring points around center (2, 2) with radius between 0.7 and 1
    phi = np.random.uniform(0, 2 * np.pi, NUM_SIGNAL_POINTS)
    radii = np.random.uniform(0.8, 1.0, NUM_SIGNAL_POINTS)
    x_ring = radii * np.cos(phi)
    y_ring = radii * np.sin(phi)
    ring_data = np.column_stack((x_ring, y_ring))*SCALE*RING_SCALE

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

# Combine datasets
data = np.vstack((ring_data, random_data))
data = np.vstack((data, ring_data*0.2))

# Shuffle the combined dataset
if SHUFFLE:
    np.random.shuffle(data)

# Instantiate the SOINN+ object
soinn = SoinnPlus(dim=DIMENSION)

# Feed the signals one by one
for point in data:
    soinn.input_signal(point)

# For now: just plot the nodes (weights)
nodes_nparray = np.vstack(soinn.nodes)
if DIMENSION == 2:
    axis = [-SCALE, SCALE, -SCALE, SCALE]
    plt.scatter(random_data[:, 0], random_data[:, 1], color='black', label=f"Noise ({NUM_NOISE_POINTS} Samples)", s=5, alpha=0.5)
    plt.scatter(ring_data[:,0], ring_data[:,1], color='blue', label=f"Signal ({NUM_SIGNAL_POINTS} Samples)", s=5, alpha=0.5)
    plt.scatter(nodes_nparray[:,0], nodes_nparray[:,1], color='red', label=f"Nodes ({len(soinn.nodes)})", s=5, alpha=0.5)
    for j, neighbors in enumerate(soinn.adjacency_mat.rows):
        for k in neighbors:
            if k >= j:  # avoid duplicate lines
                nj = nodes_nparray[j]
                nk = nodes_nparray[k]
                plt.plot([nj[0], nk[0]], [nj[1], nk[1]], 'k')
    plt.legend()
    plt.axis(axis)
    if SHUFFLE:
        plt.title("Input (shuffled)")
    else:
        plt.title("Input (sorted)")


if DIMENSION == 3:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(random_data[:,0], random_data[:,1], random_data[:,2], c='black', s=1, alpha=0.5, label=f"Noise ({NUM_NOISE_POINTS} Samples)")
    ax.scatter(ring_data[:,0], ring_data[:,1], ring_data[:,2], c='green', s=5, alpha=0.5, label=f"Signal ({NUM_SIGNAL_POINTS} Samples)")
    ax.scatter(nodes_nparray[:,0], nodes_nparray[:,1], nodes_nparray[:,2], c='r', s=20, alpha=0.9, label=f"Nodes ({len(soinn.nodes)})")

    for j, neighbors in enumerate(soinn.adjacency_mat.rows):
        for k in neighbors:
            if k >= j:  # avoid duplicate lines
                nj = nodes_nparray[j]
                nk = nodes_nparray[k]
                ax.plot([nj[0], nk[0]], [nj[1], nk[1]], [nj[2], nk[2]], 'k')

    # Optional: set aspect ratio and labels
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-SCALE*RING_SCALE, SCALE*RING_SCALE])
    ax.set_ylim([-SCALE*RING_SCALE, SCALE*RING_SCALE])
    ax.set_zlim([-SCALE*RING_SCALE, SCALE*RING_SCALE])
    ax.legend()

    if SHUFFLE:
        ax.set_title("Input (shuffled)")
    else:
        ax.set_title("Input (sorted)")

plt.show()


# soinn.show(plt_axis=axis)