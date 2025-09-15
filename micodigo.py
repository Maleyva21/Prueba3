import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=np.float32)

# Modelo secuencial con Input() y 1 capa oculta
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilación del modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping para terminar si no mejora
callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=200, restore_best_weights=True
)

# Entrenamiento con progreso visible
history = model.fit(X, y, epochs=3000, verbose=1, callbacks=[callback])

# Predicciones
preds = model.predict(X)
print("\nPredicciones finales:")
for i in range(4):
    print(f"Entrada: {X[i]} -> Salida predicha: {preds[i][0]:.4f}")

# Gráfica de la pérdida
plt.plot(history.history['loss'])
plt.xlabel('Épocas')
plt.ylabel('Loss (MSE)')
plt.title('Evolución del error (XOR)')
plt.grid(True)
plt.show()

# Frontera de decisión
h = 0.01
x_min, x_max = -0.1, 1.1
y_min, y_max = -0.1, 1.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid).reshape(xx.shape)

# Gráfica de la frontera de decisión
plt.figure(figsize=(7, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8)
for i in range(len(X)):
    color = 'ko' if y[i] == 0 else 'go'
    plt.plot(X[i, 0], X[i, 1], color, markersize=10)

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Frontera de decisión con TensorFlow/Keras')
plt.grid(True)
plt.show()
