# cnn-number-recognizer
# CNN para Classificação de Dígitos (MNIST)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

Este repositório contém um projeto completo em Python para treinar, avaliar e visualizar uma Rede Neural Convolucional (CNN) para a classificação de dígitos manuscritos do famoso dataset MNIST. O projeto foi desenvolvido com foco didático, demonstrando desde o treinamento básico até técnicas de análise e visualização de resultados.

## 📋 Tabela de Conteúdos
- [Sobre o Projeto](#sobre-o-projeto)
- [Funcionalidades](#-funcionalidades)
- [Como Começar](#-como-começar)
  - [Pré-requisitos](#pré-requisitos)
  - [Instalação](#instalação)
- [Como Usar](#-como-usar)
  - [1. Treinando o Modelo](#1-treinando-o-modelo)
  - [2. Inspecionando Previsões Individuais](#2-inspecionando-previsões-individuais)
  - [3. Analisando Todos os Resultados (Acertos e Erros)](#3-analisando-todos-os-resultados-acertos-e-erros)
- [Estrutura dos Arquivos](#-estrutura-dos-arquivos)
- [Licença](#-licença)

## 📖 Sobre o Projeto

O objetivo deste projeto é servir como um guia prático para quem está aprendendo sobre Redes Neurais Convolucionais. Através de scripts bem documentados, o usuário pode:
1.  Construir e treinar uma CNN do zero.
2.  Avaliar sua performance em um conjunto de dados não visto.
3.  Visualizar o comportamento do modelo de forma interativa.
4.  Gerar um relatório completo de todos os acertos e erros, salvando as imagens para análise posterior.

## ✨ Funcionalidades

- **Treinamento de Modelo:** Script para treinar a CNN e salvar o modelo treinado em um arquivo (`.keras`).
- **Avaliação de Performance:** Mede a acurácia e o erro (loss) no conjunto de teste.
- **Visualização Interativa:** Ferramenta para carregar o modelo e inspecionar previsões em imagens aleatórias.
- **Relatório de Erros e Acertos:** Script que processa todo o conjunto de teste e salva cada imagem em pastas organizadas (`/acertos` e `/erros`) para uma análise detalhada.
- **Data Augmentation:** Demonstração de como usar aumento de dados para tornar o modelo mais robusto (código comentado como alternativa).

## 🚀 Como Começar

Siga estas instruções para ter o projeto rodando na sua máquina local.

### Pré-requisitos

- Python 3.8 ou superior
- `pip` (gerenciador de pacotes do Python)

### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
    cd seu-repositorio
    ```

2.  **(Recomendado) Crie e ative um ambiente virtual:**
    ```bash
    # Para Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Para Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Instale as dependências:**
    Crie um arquivo chamado `requirements.txt` com o seguinte conteúdo:
    ```txt
    tensorflow
    numpy
    matplotlib
    ```
    Em seguida, instale-o via pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

A utilização do projeto é dividida em três etapas principais, correspondentes aos scripts fornecidos.

### 1. Treinando o Modelo

Para treinar a CNN pela primeira vez e gerar o arquivo `meu_modelo_cnn.keras`, execute:
```bash
python treinar_cnn.py
```
Este processo pode levar alguns minutos e irá exibir o progresso do treinamento a cada época.

### 2. Inspecionando Previsões Individuais

Após treinar e salvar o modelo, você pode inspecioná-lo de forma interativa. O script a seguir irá carregar o modelo e mostrar 5 previsões aleatórias:
```bash
python inspecionar_previsoes.py
```
A janela de visualização mostrará a imagem, a previsão do modelo e o rótulo correto.

### 3. Analisando Todos os Resultados (Acertos e Erros)

Para realizar uma análise completa e salvar todas as 10.000 imagens de teste em pastas organizadas, execute:
```bash
python salvar_todos_resultados.py
```
**Atenção:** Este script é mais longo e irá criar uma pasta chamada `resultados_mnist` com 10.000 arquivos de imagem. Após a execução, você pode navegar por esta pasta para analisar visualmente o desempenho do modelo.

## 📁 Estrutura dos Arquivos

```
.
├── treinar_cnn.py                # Script principal para treinar e salvar o modelo
├── inspecionar_previsoes.py      # Script para visualizar previsões aleatórias
├── salvar_todos_resultados.py    # Script para salvar todos os resultados em disco
├── requirements.txt              # Lista de dependências Python
├── README.md                     # Este arquivo
└── .gitignore                    # (Opcional) Para ignorar arquivos do Git
```
