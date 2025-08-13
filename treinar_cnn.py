# CÓDIGO PARA TREINAR E SALVAR O MODELO
# Execute este bloco de código completo UMA VEZ.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 1. Carregando e preparando o dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# 2. Construindo a arquitetura da CNN
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
])

# 3. Compilando o modelo
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# 4. Treinando o modelo
batch_size = 128
epochs = 15 # Pode até diminuir para 5 se quiser ser mais rápido, a acurácia já será boa.
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)

# 5. SALVANDO O MODELO TREINADO
# Esta é a linha mais importante deste bloco!
model.save('meu_modelo_cnn.keras')

print("\nModelo treinado e salvo com sucesso como 'meu_modelo_cnn.keras'!")
