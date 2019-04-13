# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import contractions
import string
import re
import nltk
from nltk.corpus import stopwords

#stopwords, punkt, wordnet
nltk.download('punkt')

poem = pd.read_csv('all.csv')
print(poem.columns)
print(poem.loc[1,'content'])

# Remove \r\n
poem['content'] = poem['content'].apply(lambda x: ' '.join(re.sub(r"^\r\n"," ",x).split()))

# Expand conatractions, remove punctuations, stopwords
poem['content'] = poem['content'].apply(lambda x: contractions.fix(x))
poem['content'] = poem['content'].apply(lambda x: x.translate(str.maketrans('','', string.punctuation)))
poem['content'] = poem['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))
stop_words = stopwords.words('english')
poem['content'] = poem['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

from nltk.tokenize import word_tokenize
poem['content'] = poem['content'].apply(lambda x: word_tokenize(x))
lemmatizer = nltk.stem.WordNetLemmatizer()

def lem(text):
    modified = ""
    for w in text:
        modified = modified + " " +lemmatizer.lemmatize(w)
    return modified

poem['content'] = poem['content'].apply(lambda x: lem(x))
print(poem.groupby('type').count())

X_train = poem.loc[:410, 'content'].values
y_train = poem.loc[:410, 'type'].values
X_test = poem.loc[411:, 'content'].values
y_test = poem.loc[411:, 'type'].values

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(train_vectors, y_train)

from  sklearn.metrics  import accuracy_score
predicted = clf.predict(test_vectors)
print(accuracy_score(y_test,predicted))

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Conv1D, MaxPooling1D, BatchNormalization 
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

max_features = 4000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(poem['content'].values)
X = tokenizer.texts_to_sequences(poem['content'].values)
X = pad_sequences(X)

Y = pd.get_dummies(poem['type']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.35, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

#Download Glove Word Embeddings (https://nlp.stanford.edu/projects/glove/),
#from the glove.6B.zip folder extract glove.6B.300d text file

embeddings_index = dict()
f = open('glove.6B.300d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

vocabulary_size = 5000
embedding_matrix = np.zeros((vocabulary_size, 300))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
            
model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 300, input_length=X.shape[1], weights=[embedding_matrix], trainable=False))
model_glove.add(Conv1D(64, 5, activation='relu'))
model_glove.add(MaxPooling1D(pool_size=4))
model_glove.add(LSTM(100))
model_glove.add(Dense(3,activation='softmax'))
model_glove.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model_glove.summary())

batch_size = 16
model_glove.fit(X_train, Y_train, epochs = 8, batch_size=batch_size, verbose = 2)

#model_glove.load_weights('glovelstm.h5')
y = model_glove.predict(X_test)
predicted = [yy.argmax(axis=0) for yy in y]
actual = [yy.argmax(axis=0) for yy in Y_test]

count = 0
for i in range(len(predicted)):
    if predicted[i] == actual[i]:
        count = count+1
print(count)
print(count/len(predicted))

model_glove.save_weights("glovelstm.h5")
print("Saved model to disk")

## Plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
from sklearn.manifold import TSNE

## Get weights
conv_embds = model_glove.layers[0].get_weights()[0]

word_list = []
for word, i in tokenizer.word_index.items():
    word_list.append(word)
    
## Plotting function
def plot_words(data, start, stop, step):
    trace = go.Scatter(
        x = data[start:stop:step,0], 
        y = data[start:stop:step, 1],
        mode = 'markers',
        text= word_list[start:stop:step]
    )
    layout = dict(title= 't-SNE 1 vs t-SNE 2',
                  yaxis = dict(title='t-SNE 2'),
                  xaxis = dict(title='t-SNE 1'),
                  hovermode= 'closest')
    fig = dict(data = [trace], layout= layout)
    py.iplot(fig)

conv_tsne_embds = TSNE(n_components=2).fit_transform(conv_embds)
plot_words(conv_tsne_embds, 0, 2000, 1)



