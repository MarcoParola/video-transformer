import matplotlib.pyplot as plt
import numpy as np



trainable_params = {
    'detr-b': 46580465,
    'detr-t': 19272433,
    'concat-detr-b': 46832177,
    'yolos-t': 6383241,
    'concat-yolos-t': 6501321,
    'yolos-b': 42584329,
    'affine-detr-b': 49809457,
}


overhead_concat = trainable_params['concat-detr-b'] - trainable_params['detr-b']
overhead_affine = trainable_params['affine-detr-b'] - trainable_params['detr-b']


# plot bar chart of trainable parameters
# the plot is composed of 4 groups (detr-b, yolos-b, detr-t, yolos-t), each of them has 3 bars (base, concat, affine)
barWidth = 0.25
fig, ax = plt.subplots()
groups = 4
r1 = np.arange(groups)
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

ax.bar(r1, [trainable_params['detr-t'], trainable_params['detr-b'], trainable_params['yolos-t'], trainable_params['yolos-b']], width=barWidth, label='Base', alpha=0.8)
ax.bar(r2, [overhead_concat + trainable_params['detr-t'], overhead_concat + trainable_params['detr-b'], overhead_concat + trainable_params['yolos-t'], overhead_concat + trainable_params['yolos-b']], width=barWidth, label='Concat', alpha=0.8)
ax.bar(r3, [overhead_affine + trainable_params['detr-t'], overhead_affine + trainable_params['detr-b'], overhead_affine + trainable_params['yolos-t'], overhead_affine + trainable_params['yolos-b']], width=barWidth, label='Affine', alpha=0.8)

x_labels = ['DETR-t', 'DETR-b', 'YOLOS-t', 'YOLOS-b']
ax.set_xticks([r + barWidth for r in range(groups)])
ax.set_xticklabels(x_labels)
ax.set_ylabel('Trainable Parameters')
ax.legend()
plt.show()
