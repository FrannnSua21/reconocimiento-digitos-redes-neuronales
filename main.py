import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. Cargar datos
(x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = tf.keras.datasets.mnist.load_data()

# 2. Preprocesar: normalizar a [0,1] y añadir canal (28, 28, 1)
x_entrenamiento = x_entrenamiento[..., np.newaxis] / 255.0
x_prueba = x_prueba[..., np.newaxis] / 255.0

# 3. Modelo CNN
modelo = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation="softmax"),
])

modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 4. Entrenar
modelo.fit(x_entrenamiento, y_entrenamiento, epochs=5, batch_size=128, validation_split=0.1)

# 5. Evaluar
perdida, precision = modelo.evaluate(x_prueba, y_prueba, verbose=0)
print(f"Precisión en prueba: {precision:.4f}")

# 6. Predecir y mostrar ejemplos
predicciones = modelo.predict(x_prueba[:10])
fig, ejes = plt.subplots(2, 5, figsize=(10, 5))
for i, eje in enumerate(ejes.flat):
    eje.imshow(x_prueba[i].squeeze(), cmap="gray")
    eje.set_title(f"Pred: {predicciones[i].argmax()} | Real: {y_prueba[i]}")
    eje.axis("off")
plt.tight_layout()
plt.show()

# 7. Guardar el modelo
modelo.save("modelo_mnist.keras")