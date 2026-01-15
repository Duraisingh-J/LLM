import numpy as np

sentences = [
    "I love deep learning",
    "I love NLP",
    "deep learning loves data"
]

embedding_dim = 5


words = " ".join(sentences).lower().split()
vocab = list(set(words))

word2idx = {word : i for i, word in enumerate(vocab) }
idx2word = {i : word for word, i in word2idx.items() }

# Scores into Probabilities 
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# One Hot Encoding 
vocab_size = len(vocab)

W1 = np.random.randn(vocab_size, embedding_dim)
W2 = np.random.randn(embedding_dim, vocab_size)

def one_hot(word):
    vec = np.zeros(vocab_size)
    vec[word2idx[word]] = 1
    return vec

# CBOW training pairs 

window_size = 1 # one word left, one right 
data = []

for sentence in sentences:
    tokens = sentence.lower().split()
    for i in range(window_size, len(tokens) - window_size):
        context = [
            tokens[i - 1],
            tokens[i + 1]
        ]

        target = tokens[i]
        data.append((context, target))


# Training Phase
lr = 0.01
epochs = 1000

for epoch in range(epochs):
    loss = 0

    for context, target in data:
        context_vecs = np.array([one_hot(w) for w in context])
        x = np.mean(context_vecs, axis=0) # CBOW averages context

        # Forward pass
        h = np.dot(x, W1) # Hidden layer
        o = np.dot(h, W2) # Output scores
        y_pred = softmax(o)

        y_true = one_hot(target)

        # Loss
        loss += -np.sum(y_true * np.log(y_pred + 1e-9))

        # Backpropagation
        error = y_pred - y_true

        dW2 = np.outer(h, error)
        dW1 = np.outer(x, np.dot(W2, error))

        # Update weights
        W1 -= lr * dW1
        W2 -= lr * dW2

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss : {loss : .4f}")


# def get_embedding(word):
#     return W1[word2idx[word]]

# print("Embedding for 'deep': ", get_embedding("deep"))
# print("Embedding for 'learning' : ", get_embedding("learning"))


def predict_word(context_words):

    context_vecs = np.array([one_hot(w) for w in context_words])
    x = np.mean(context_vecs, axis=0)

    h = np.dot(x, W1)
    u = np.dot(h, W2)
    y_pred = softmax(u)

    predicted_idx = np.argmax(y_pred)
    return idx2word[predicted_idx]

print(predict_word(["i", "learning"]))       # should predict "love"
print(predict_word(["love", "learning"])) # should predict "deep"
print(predict_word(["deep", "loves"]))    # should predict "learning"
