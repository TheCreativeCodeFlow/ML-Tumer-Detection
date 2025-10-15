# ------------------------------------------------------------
# Figure 3: Multi-Dimensional Radar Visualization
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# Metrics and models
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC', 'AUC']
models = ['BERT', 'RoBERTa', 'DeBERTa', 'ELECTRA', 'DistilBERT', 'N-gram']

data = {
    'BERT':      [99.5, 99.2, 99.8, 99.5, 98.9, 99.8],
    'RoBERTa':   [99.3, 99.0, 99.6, 99.3, 98.6, 99.7],
    'DeBERTa':   [99.7, 99.4, 99.9, 99.6, 99.3, 99.9],
    'ELECTRA':   [98.8, 98.5, 99.0, 98.7, 97.9, 99.5],
    'DistilBERT':[97.9, 97.4, 98.2, 97.8, 96.8, 99.2],
    'N-gram':    [72.8,6 71.9, 73.4, 72.6, 68.3, 79.4]
}

# Prepare angles
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # close the loop

# Create radar plot
plt.figure(figsize=(8,8))
ax = plt.subplot(111, polar=True)

for model in models[:5]:  # plot transformers only for clarity
    values = np.array(data[model]) / 100  # normalize to 0-1
    values = np.append(values, values[0]) # close shape
    ax.plot(angles, values, linewidth=2, label=model)
    ax.fill(angles, values, alpha=0.08)

# Customization
ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
ax.set_title('Figure 3: Multi-Dimensional Radar Chart', pad=20)
ax.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()
plt.show()
