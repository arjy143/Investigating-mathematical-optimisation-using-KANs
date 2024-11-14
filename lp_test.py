import tensorflow as tf
from keras import layers, models
import numpy as np

# Define the neural network model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(2,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='linear')  # 2 outputs, for x1 and x2
])

# Define the custom loss function to incorporate constraints
def lp_loss(y_true, y_pred):
    x1, x2 = y_pred[:, 0], y_pred[:, 1]

    # Objective function: -maximize 3*x1 + 5*x2
    objective = -1 * (3 * x1 + 5 * x2) 

    # Define penalties for constraints
    constraint1 = tf.maximum(12.0, 2 * x1 + 3 * x2)  # 2*x1 + 3*x2 <= 12
    constraint2 = tf.maximum(8.0, 2 * x1 + x2)       # 2*x1 + x2 <= 8
    constraint3 = tf.maximum(0.0, -x1)                   # x1 >= 0
    constraint4 = tf.maximum(0.0, -x2)                   # x2 >= 0

    # Sum up the objective and constraint penalties
    loss = objective + 100 * (constraint1 + constraint2 + constraint3 + constraint4)
    return loss

# Compile the model with a low learning rate for gradual optimization
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=lp_loss)

# Training data: we just need any random input as we are not using it (dummy data)
x_train = np.random.rand(100, 2)
y_train = np.zeros((100, 2))

# Train the model
history = model.fit(x_train, y_train, epochs=500, verbose=0)

# Predict the solution
solution = model.predict(np.array([[0.5, 0.5]]))  # Provide some input, irrelevant here

x1, x2 = solution[0]
objective_value = 3 * x1 + 5 * x2

print("Optimized values: x1 =", x1, ", x2 =", x2)
print("Objective value:", objective_value)
