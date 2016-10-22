import matplotlib.pyplot as plt
import numpy as np

x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(x_data,y_data)
plt.show()