# Vector Embedding Project

This project implements a Word2Vec model using a small corpus of sentences. It includes functions for Adagrad optimization and visualizes word embeddings using PCA.

## Purpose

The main goal of this project is to demonstrate how to train a Word2Vec model on a small dataset, perform optimization using the Adagrad algorithm, and visualize the resulting word embeddings in a two-dimensional space.

## Files

- `vector_embedding.py`: Contains the implementation of the Word2Vec model, Adagrad optimization, and PCA visualization of word embeddings.

## Usage

1. **Environment Setup**:
   - Ensure you have Python installed on your machine.
   - Install the required packages using pip:
     ```
     pip install numpy gensim scikit-learn matplotlib
     ```

2. **Running the Code**:
   - Execute the `vector_embedding.py` script to train the Word2Vec model, perform optimization, and visualize the embeddings:
     ```
     python vector_embedding.py
     ```

3. **Output**:
   - The script will print the results of the Adagrad optimization steps and display a plot of the word embeddings reduced to two dimensions using PCA.

## License

This project is licensed under the MIT License - see the LICENSE file for details.