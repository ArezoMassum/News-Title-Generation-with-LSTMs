# News-Title-Generation-with-LSTMs
# News Title Generation with LSTMs

## Overview
This repository implements a text generation model using LSTM networks to generate political news titles based on the **News Category Dataset**. The goal is to train an LSTM-based model to minimize perplexity while generating credible and coherent news headlines.

## Features

- **Dataset Handling**: Loads and preprocesses the **News Category Dataset**, filtering only **POLITICS** headlines.
- **Tokenization & Vocabulary Creation**: Converts headlines to lowercase, tokenizes at the word level, and builds word-to-index mappings.
- **Custom LSTM Model**: Implements an LSTM-based text generation model with an embedding layer, stacked LSTMs, dropout, and fully connected layers.
- **Training & Optimization**: Implements a training loop with loss computation, gradient clipping, and perplexity tracking.
- **Sentence Generation Strategies**: Supports **random sampling** and **greedy sampling** for generating headlines.
- **Truncated Backpropagation Through Time (TBBTT)**: Implements TBBTT to improve training efficiency and handle long sequences.
- **Loss & Perplexity Visualization**: Tracks loss reduction and perplexity improvement over epochs.



##  Model Architecture

The implemented LSTM model consists of:

- **Word Embedding Layer**
- **Stacked LSTM Layers**
- **Dropout Regularization**
- **Fully Connected Output Layer**


