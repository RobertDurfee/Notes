import numpy as np
import matplotlib.pyplot as plt

name = 'Robert Durfee'

x = np.linspace(0, 360)
y = np.sin(x*np.pi/180.0)

plt.plot(x,y)
plt.xlabel('degrees')
plt.ylabel('sine')
plt.title('{} has the required software!'.format(name))
plt.show()