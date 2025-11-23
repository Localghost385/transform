import matplotlib.pyplot as plt
import numpy as np

# Adaptive weights
adaptive_weights = np.array([
    [0.21768469, 0.5388223,  0.243493],
    [0.21538684, 0.55666524, 0.22794795],
    [0.21489023, 0.5488037,  0.23630609],
    [0.21669662, 0.5491201,  0.23418328],
    [0.21985134, 0.5093217,  0.27082705],
    [0.21514185, 0.55847436, 0.22638386],
    [0.21739748, 0.5222093,  0.2603932],
    [0.21832526, 0.5235855,  0.25808924],
    [0.21656644, 0.542948,   0.2404856],
    [0.21573794, 0.5031109,  0.2811512],
    [0.2152038,  0.5590442,  0.22575206],
    [0.21671237, 0.5432577,  0.24002993],
    [0.21761899, 0.51635474, 0.2660263],
    [0.21478905, 0.5616469,  0.22356407],
    [0.21777475, 0.5362841,  0.24594119],
    [0.21684882, 0.5429876,  0.24016364],
    [0.21636812, 0.5513364,  0.23229548],
    [0.21813008, 0.50502825, 0.27684167],
    [0.2198277,  0.5090722,  0.2711001],
    [0.21603255, 0.5546061,  0.22936134],
    [0.21405028, 0.5638119,  0.22213785]
])

# Loss values
loss_vals = np.array([
    0.271170, 0.131308, 0.110422, 0.104814, 0.099013, 0.090547, 0.086497,
    0.085535, 0.085373, 0.084544, 0.084218, 0.083934, 0.083484, 0.083377,
    0.082853, 0.082861, 0.082637, 0.082240, 0.081891, 0.081948, 0.081777
])

epochs = np.arange(len(loss_vals))

fig, ax1 = plt.subplots(figsize=(12,5))

# Plot loss
ax1.plot(epochs, loss_vals, 'k-o', label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss and Adaptive Weights over Epochs')
ax1.grid(True)

# Plot adaptive weights on same figure with second y-axis
ax2 = ax1.twinx()
ax2.plot(epochs, adaptive_weights[:,0], 'r--', label='Dense Fraction')
ax2.plot(epochs, adaptive_weights[:,1], 'g--', label='Sync')
ax2.plot(epochs, adaptive_weights[:,2], 'b--', label='IOI Variance')
ax2.set_ylabel('Adaptive Weight')

# Legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.show()
