# V1 — Statistical Bigram Language Model

## Overview

This is the first version of my `makemore-from-scratch` project, inspired by Andrej Karpathy's *Neural Networks: Zero to Hero* series.

The goal is to build a character-level name generator using statistical methods, without neural networks or gradient-based optimization.

The model learns character-transition probabilities from a dataset of names and uses them to generate new names.

## How It Works

### 1. Character Vocabulary

The dataset contains names made up of 26 lowercase English letters. I introduce a special token, `.`, to represent both the beginning and end of a name.

This gives us a vocabulary of 27 characters.

I create two mappings:

- `stoi`: Maps each character to an integer.
- `itos`: Maps each integer back to its character.

### 2. Bigram Count Matrix

A bigram is a pair of consecutive characters.

For example, the name `emma` becomes:

` . → e → m → m → a → . `

I construct a 27 × 27 count matrix, `N`, where:

- Rows represent the current character.
- Columns represent the next character.
- Each cell stores the number of times that transition occurs in the dataset.

This matrix captures the statistical patterns of consecutive characters in names.

### 3. Transition Probabilities

Raw counts tell us how frequently transitions occur. To turn these counts into probabilities, I normalize each row by its sum.

Each row then represents a probability distribution over the possible next characters.

### 4. Laplace Smoothing

Some character transitions may never occur in the training dataset, resulting in zero probabilities.

To address this, I add one to every count before normalizing the matrix. This technique is called **Laplace smoothing** or **add-one smoothing**.

It ensures every possible transition has a nonzero probability and every row can be normalized into a valid probability distribution.

### 5. Name Generation

Starting with the `.` token, the model repeatedly samples the next character using `torch.multinomial()` and the probability distribution corresponding to the current character.

The process continues until the model generates `.` again, marking the end of the name.

A seeded random generator makes the sampling reproducible.

## Model Evaluation — Negative Log-Likelihood

To evaluate the model, I calculate the average negative log-likelihood (NLL) over the character transitions in the dataset.

For each observed bigram, I retrieve its predicted probability, take its logarithm, and negate the average across all observed transitions.

\[
\text{NLL}=-\frac{1}{N}\sum_{i=1}^{N}\log P(y_i\mid x_i)
\]

``` Current loss : 2.4543561935424805 ```

Here, \(N\) is the total number of observed bigrams, \(x_i\) is the current character, and \(y_i\) is the next character.

A lower NLL indicates that the model assigns higher probabilities to the observed transitions. It is useful for comparing successive model versions when evaluated on the same data.

## What I Learned

- Representing text as numerical data using a character vocabulary.
- Modeling character transitions using a count matrix.
- Converting frequencies into probability distributions.
- Handling unseen transitions using Laplace smoothing.
- Generating sequences through probabilistic sampling.
- Evaluating a language model using negative log-likelihood.

## Limitations

- The model considers only the immediately preceding character.
- It cannot directly learn longer character patterns or dependencies.
- Its behavior is determined entirely by observed transition counts and the smoothing strategy.
- It does not learn parameters through gradient descent.

## What's Next?

**V2 — Neural Bigram Language Model**

The next version will model the same character transitions using a neural network trained with gradient-based optimization.

This will let me compare a statistical approach with a learnable model while building toward a multilayer perceptron (MLP).

---

*Part of my `makemore-from-scratch` learning journey, building language models step by step to understand the underlying concepts of deep learning.*