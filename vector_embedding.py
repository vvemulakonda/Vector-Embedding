import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Define a small corpus of sentences as a list of tokenized words
corpus = [
    ["quantum", "computing", "promises", "exponential", "speedup", "over", "classical", "algorithms"],
    ["neural", "networks", "excel", "at", "image", "classification", "tasks"],
    ["the", "singularity", "is", "a", "hypothetical", "point", "when", "machine", "intelligence", "surpasses", "human"],
    ["black", "holes", "emit", "hawking", "radiation", "according", "to", "theoretical", "predictions"],
    ["genome", "editing", "with", "crispr", "has", "revolutionized", "biological", "research"],
    ["large", "language", "models", "have", "transformative", "potential", "in", "natural", "language", "processing"],
    ["data", "privacy", "regulations", "like", "gdpr", "protect", "user", "information"],
    ["carbon", "capture", "technology", "is", "being", "developed", "to", "combat", "climate", "change"]
] 

# Train a Word2Vec model on the corpus
model = Word2Vec(sentences=corpus, vector_size=20, window=3, min_count=1, workers=1)

# Print semantic relationships between selected words
print("Similarity between 'quantum' and 'neural':", model.wv.similarity("quantum", "neural"))
print("Similarity between 'machine' and 'human':", model.wv.similarity("machine", "human"))
print("Most similar words to 'crispr':", model.wv.most_similar("crispr", topn=3))

# Select a word and print its embedding
word = "quantum"
embedding = model.wv[word]
print(f"Vector embedding for '{word}':\n{embedding}\n")

# Normalize the embedding and print it
embedding_norm = normalize(embedding.reshape(1, -1))
print(f"Normalized embedding for '{word}':\n{embedding_norm}\n")

# Define a non-linear, and non-convex function for ADAgrad optimization
def f(x):
    return np.sin(x) + 0.05 * x ** 3 - 0.5 * x
# Define the gradient (derivative) of the function
def grad_f(x):
    return np.cos(x) + 0.15 * x ** 2 - 0.5

# Initialization for Adagrad optimization
x = 0.0
learning_rate = 1.0
eps = 1e-8
gradient_sum = 0.0
# Perform Adagrad optimization for 10 steps
print("Adagrad Optimization Steps:")
for i in range(10):
    g = grad_f(x)
    gradient_sum += g ** 2
    adjusted_lr = learning_rate / (np.sqrt(gradient_sum) + eps)  # Learning rate adjustment
    x -= adjusted_lr * g
    print(f"Step {i+1}: x = {x}, f(x) = {f(x)}, gradient = {g}, adjusted_lr = {adjusted_lr}")

#Get all word embeddings from the model
embeddings = np.array([model.wv[word] for word in model.wv.index_tokey])
# Perform PCA to reduce dimensionality for visualization
#Use 2 components for 2D visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)
# Plot the 2D embeddings
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o')
for i, word in enumerate(model.wv.index_to_key):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=12)
plt.title("Word Embeddings Visualized with PCA")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid()
plt.show()