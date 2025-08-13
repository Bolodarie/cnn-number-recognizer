# -*- coding: utf-8 -*-
"""
Script para processar TODO o conjunto de teste de um modelo CNN treinado para o MNIST.
Ele salva cada imagem classificada em uma pasta de 'acertos' ou 'erros'.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# --- CONFIGURAÇÃO E PREPARAÇÃO ---

# Nome do arquivo do modelo salvo
MODELO_ARQUIVO = 'meu_modelo_cnn.keras'

# Nome da pasta onde os resultados serão salvos
PASTA_SAIDA = 'resultados_mnist'

# 1. Checar se o modelo existe antes de começar
if not os.path.exists(MODELO_ARQUIVO):
    print(f"Erro: Arquivo do modelo '{MODELO_ARQUIVO}' não encontrado.")
    print("Por favor, execute primeiro o script de treinamento para gerar este arquivo.")
    exit()

# 2. Carregar o modelo treinado
print(f"Carregando modelo de '{MODELO_ARQUIVO}'...")
model = keras.models.load_model(MODELO_ARQUIVO)
print("Modelo carregado com sucesso.")

# 3. Carregar os dados de teste
(x_train, y_train), (x_test_imgs, y_test_labels) = keras.datasets.mnist.load_data()

# 4. Pré-processar as imagens de teste para o modelo
print("Pré-processando imagens de teste...")
x_test_proc = x_test_imgs.astype("float32") / 255.0
x_test_proc = np.expand_dims(x_test_proc, -1)

# --- EXECUÇÃO PRINCIPAL ---

# 5. Fazer previsões para todo o conjunto de teste
print(f"Iniciando previsões para as {len(x_test_proc)} imagens de teste...")
start_time = time.time()
predictions_array = model.predict(x_test_proc)
predicted_labels = np.argmax(predictions_array, axis=1)
end_time = time.time()
print(f"Previsões concluídas em {end_time - start_time:.2f} segundos.")

# 6. Criar as pastas de saída
print(f"Criando estrutura de pastas em '{PASTA_SAIDA}'...")
os.makedirs(os.path.join(PASTA_SAIDA, 'erros'), exist_ok=True)
for i in range(10):
    os.makedirs(os.path.join(PASTA_SAIDA, 'acertos', str(i)), exist_ok=True)
print("Estrutura de pastas pronta.")

# --- SALVANDO OS RESULTADOS ---

# 7. Loop para salvar cada imagem no local correto
print("\nIniciando processo de salvar todas as imagens...")
print("Isso pode levar vários minutos. Aguarde...")

num_erros = 0
num_acertos = 0

for i in range(len(y_test_labels)):
    # Pega os dados da imagem atual
    imagem = x_test_imgs[i]
    label_correto = y_test_labels[i]
    label_previsto = predicted_labels[i]

    # Define o título e a cor para a imagem
    if label_previsto == label_correto:
        num_acertos += 1
        subpasta = os.path.join('acertos', str(label_correto))
        titulo = f"Acerto: {label_previsto}"
        cor = 'green'
    else:
        num_erros += 1
        subpasta = 'erros'
        titulo = f"Prev: {label_previsto}, Certo: {label_correto}"
        cor = 'red'

    # Monta o caminho completo do arquivo
    nome_arquivo = f"img_{i}_prev_{label_previsto}_certo_{label_correto}.png"
    caminho_completo = os.path.join(PASTA_SAIDA, subpasta, nome_arquivo)

    # Cria e salva a figura
    fig = plt.figure(figsize=(3, 3))
    plt.imshow(imagem, cmap=plt.cm.binary)
    plt.title(titulo, color=cor)
    plt.xticks([])
    plt.yticks([])
    
    # Salva a figura no disco
    plt.savefig(caminho_completo)
    
    # IMPORTANTE: Fecha a figura para liberar memória
    plt.close(fig)

    # Imprime um status de progresso para o usuário
    if (i + 1) % 500 == 0:
        print(f"  ... {i + 1} de {len(y_test_labels)} imagens processadas.")

# --- CONCLUSÃO ---

print("\n--- Processo Concluído! ---")
print(f"Total de imagens salvas: {len(y_test_labels)}")
print(f"  - Acertos: {num_acertos} (salvos na pasta '{os.path.join(PASTA_SAIDA, 'acertos')}')")
print(f"  - Erros: {num_erros} (salvos na pasta '{os.path.join(PASTA_SAIDA, 'erros')}')")
print("\nVocê já pode navegar pelas pastas para ver os resultados.")
