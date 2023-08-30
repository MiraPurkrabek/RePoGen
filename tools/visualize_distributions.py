import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

LOWER_BOUND = -40
MODE = 0
UPPER_BOUND = 120

# Set all plt text to bold
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# Set all plt axes to thick
plt.rcParams["axes.linewidth"] = 2.0

x = np.linspace(LOWER_BOUND-10, UPPER_BOUND+10, 1000)
x1 = np.linspace(LOWER_BOUND-10, MODE, 500)
x2 = np.linspace(MODE, UPPER_BOUND+10, 500)

y_uniform = stats.uniform.pdf(x, loc=LOWER_BOUND, scale=UPPER_BOUND-LOWER_BOUND)

y_skewed_gauss1 = stats.norm.pdf(x1, loc=MODE, scale=0.5*(MODE-LOWER_BOUND))
y_skewed_gauss2 = stats.norm.pdf(x2, loc=MODE, scale=0.5*(UPPER_BOUND-MODE))
y_skewed_gauss = np.concatenate((y_skewed_gauss1, y_skewed_gauss2))

further_bound = max(abs(LOWER_BOUND-MODE), abs(UPPER_BOUND-MODE)) / 1.5
scale = further_bound / 2.0
y_truncated_gauss = stats.norm.pdf(x, loc=MODE, scale=scale)

plt.plot(np.concatenate([x1, x2]), y_skewed_gauss, label='Skewed Gaussian', color='b', linewidth=2)
plt.plot(x, y_uniform, label='Uniform', color='r', linewidth=2)
plt.plot(x, y_truncated_gauss, label='Truncated Gaussian', color='orange', linewidth=2)

plt.axvline(x=MODE, color='k', linestyle='--')
plt.axvline(x=LOWER_BOUND, color='g', linestyle='--')
plt.axvline(x=UPPER_BOUND, color='g', linestyle='--')
plt.xlabel('Angle (degrees)')
plt.legend(["Skewed Gaussian", "Uniform", "Truncated Gaussian", "Mode", "Limits"], loc='upper right')
plt.xlim([LOWER_BOUND-20, UPPER_BOUND+20])
plt.grid(False)
plt.tick_params(left=False, labelleft=False)
plt.show()
