import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# X Data
N = 200
X = np.random.random(N)
# Generate Y Data
sign = (- np.ones((N,)))**np.random.randint(2,size=N)
Y = np.sqrt(X)*sign
#print (y)

# Neural Network
act = tf.keras.layers.ReLU()
nn_sv = tf.keras.models.Sequential([tf.keras.layers.Dense(10, activation = act),
                                    tf.keras.layers.Dense(10, activation = act),
                                    tf.keras.layers.Dense(1, activation = 'linear')])
# Loss function
loss_sv = tf.keras.losses.MeanSquaredError()
def loss_dp (y_true, y_pred):
    return loss_sv(y_true, y_pred**2)

optim_sv = tf.keras.optimizers.Adam(lr = 0.001)

# Compile
#nn_sv.compile(optimizer = optim_sv, loss = loss_sv)
nn_sv.compile(optimizer = optim_sv, loss = loss_dp)

# Training
train_sv = nn_sv.fit(X, X, epochs = 5, batch_size = 5, verbose = 1)

# Plotting
plt.plot(X, Y, '.', label = 'Input Data', color = 'lightgray')
plt.plot(X, nn_sv.predict(X), '.', label = 'ML', color = 'red')
plt.xlabel('Y')
plt.ylabel ('X')
plt.title('Data Generation')
plt.legend
plt.show()