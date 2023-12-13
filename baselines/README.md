# Augmented Property Predictor

The `AugmentedPropertyPredictor` is a neural network module designed for predicting various properties of proteins. It is capable of incorporating additional features and using different types of embeddings and model architectures to make predictions.

## Overview

This module is part of a larger framework aimed at understanding and predicting protein behavior. The model is flexible and supports various configurations for embeddings, prediction heads, and additional features. It can be used for both supervised and unsupervised learning tasks.

## Features

- **Embedding Support**: The model supports various types of embeddings including pre-trained models like MSA Transformer, ESM1v, and Tranception, as well as simpler embeddings like linear embeddings and one-hot encoding.
- **Model Architectures**: Multiple architectures are supported for the prediction head, including Multi-Layer Perceptron (MLP), ConvBERT, CNN, and light attention mechanisms.
- **Uncertainty Estimation**: The model can estimate uncertainty in its predictions using Monte Carlo dropout.
- **Flexible Input**: It accepts protein sequences in the form of tokens and can handle additional inputs like unsupervised fitness predictions and precomputed sequence embeddings.
- **Customizable**: The model can be configured to predict different protein properties by adjusting the target heads and input dimensions.

## Usage

The model requires a configuration object (`args`) and an alphabet object for tokenization. The configuration object includes parameters for the model type, embedding type, dropout rates, and other hyperparameters. The alphabet object defines the tokenization scheme and special tokens like padding, mask, and end-of-sequence markers.

### Initialization

```python
model = AugmentedPropertyPredictor(args, alphabet)
```

### Forward Pass

To make predictions, a forward pass is performed with the input tokens:

```python
result = model.forward(tokens)
```

For uncertainty estimation, multiple forward passes with dropout can be used:

```python
uncertainty_result = model.forward_with_uncertainty(tokens, num_MC_dropout_samples=10)
```

### Loss Computation

The model can calculate the loss between its predictions and true labels, which is useful for training:

```python
loss, loss_dict = model.prediction_loss(target_predictions, target_labels)
```

### Optimizer

An optimizer can be created for the model, which is used during the training process:

```python
optimizer = model.create_optimizer()
```

## Model Components

### Embedding Layer

The embedding layer converts input tokens into dense vectors. It can utilize pre-trained models or simpler embeddings based on the configuration.

### Pre-Head Layer

Before making predictions, the model can apply additional layers like MLP, ConvBERT, CNN, or attention mechanisms to process the embeddings further.

### Prediction Heads

For each target property, the model has a separate prediction head. These heads are linear layers that map the processed embeddings to the desired output space.

### Uncertainty Estimation

The model can perform Monte Carlo dropout during inference to estimate the uncertainty of its predictions.

## Configuration

The model's behavior is highly dependent on the configuration provided during initialization. Users can specify the type of embeddings, the architecture of the prediction head, the number of attention heads, the dropout rate, and other parameters.

## Dependencies

The model relies on several external libraries and modules, including PyTorch, Huggingface's Transformers, and custom utility modules for protein sequence modeling.
