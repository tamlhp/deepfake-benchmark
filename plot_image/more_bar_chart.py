import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 5
meso4 = (0.5, 0.65, 0.84, 0.7,0.51)
capsule = (0.84,	0.89,	0.96,	0.95,	0.88)
xception = (0.93,	0.97,	0.98,	0.95,	0.88)
gan = (0.72,	0.73,	0.86,	0.86,	0.72)
spectrum = (0.81,	0.83,	0.98,	0.67,	0.57)
headpose = (0.64,	0.64,	0.64,	0.64,	0.62)
visual = (0.96,	0.96,	0.97,	0.84,	0.69)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.barh(index, meso4, bar_width,
alpha=opacity,
color='b',
label='meso4')

rects2 = plt.barh(index + bar_width, capsule, bar_width,
alpha=opacity,
color='g',
label='capsule')

rects3 = plt.barh(index + 2*bar_width, capsule, bar_width,
alpha=opacity,
color='r',
label='xception')

# plt.xlabel('Person')
# plt.ylabel('Scores')
plt.title('Scores by person')
plt.yticks(index + bar_width, ('meso4', 'capsule', 'xception', 'gan'))
plt.legend()

plt.tight_layout()
plt.show()