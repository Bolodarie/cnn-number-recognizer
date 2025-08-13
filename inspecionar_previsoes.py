# CÓDIGO PARA INSPECIONAR AS PREVISÕES, IMAGEM POR IMAGEM

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# 1. Carregar o modelo que salvamos anteriormente
print("Carregando modelo treinado...")
model = keras.models.load_model('meu_modelo_cnn.keras')
print("Modelo carregado!")

# 2. Carregar apenas os dados de TESTE do MNIST
# Não precisamos dos dados de treino aqui.
(_, _), (x_test_imgs, y_test_labels) = keras.datasets.mnist.load_data()

# 3. Função para fazer e mostrar a previsão
def inspecionar_previsao(index):
    # Pega a imagem e o rótulo correto do conjunto de teste
    img = x_test_imgs[index]
    label_correto = y_test_labels[index]

    # Prepara a imagem para o modelo
    # O modelo espera um "lote" de imagens, então criamos um lote com uma imagem só.
    # Também normalizamos e adicionamos a dimensão de canal, EXATAMENTE como no treino.
    img_para_prever = np.expand_dims(img, axis=0)
    img_para_prever = np.expand_dims(img_para_prever, axis=-1)
    img_para_prever = img_para_prever.astype("float32") / 255.0

    # Faz a previsão
    previsao_array = model.predict(img_para_prever)
    
    # A previsão é um array com 10 probabilidades. Pegamos o índice do maior valor.
    previsao_digito = np.argmax(previsao_array)

    # Mostra o resultado
    plt.imshow(img, cmap=plt.cm.binary)
    titulo = f"Previsão: {previsao_digito} | Correto: {label_correto}"
    
    # Define a cor do título: verde para acerto, vermelho para erro
    cor = 'green' if previsao_digito == label_correto else 'red'
    plt.title(titulo, color=cor)
    plt.show()

    # Mostra o array de confiança do modelo (opcional)
    print(f"Confiança da Previsão: {previsao_array[0][previsao_digito]*100:.2f}%")
    print("Confiança para cada dígito:", np.round(previsao_array[0], 2))


# 4. Loop para inspecionar várias imagens aleatórias
# Rode este bloco quantas vezes quiser para ver novos exemplos.
for _ in range(5): # Vamos inspecionar 5 imagens
    # Escolhe um índice aleatório do conjunto de teste
    indice_aleatorio = np.random.randint(0, len(x_test_imgs))
    inspecionar_previsao(indice_aleatorio)
