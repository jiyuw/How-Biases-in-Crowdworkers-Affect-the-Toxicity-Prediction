import numpy as np
import pandas as pd
from tqdm import tqdm

tqdm.pandas()
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
from keras.callbacks import LearningRateScheduler
import os
from gensim.models import KeyedVectors

# crawl-300d-2M.vec--> https://fasttext.cc/docs/en/english-vectors.html
# When pre-train embedding is helpful? https://www.aclweb.org/anthology/N18-2084
# There are many pretrained word embedding models:
# fasttext, GloVe, Word2Vec, etc
# crawl-300d-2M.vec is trained from Common Crawl (a website that collects almost everything)
# it has 2 million words. Each word is represent by a vector of 300 dimensions.

# https://nlp.stanford.edu/projects/glove/
# GloVe is similar to crawl-300d-2M.vec. Probably, they use different algorithms.
# glove.840B.300d.zip: Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)
# tokens mean words. It has 2.2M different words and 840B (likely duplicated) words in total

# note that these two pre-trained models give 300d vectors.
EMBEDDING_FILES = [
    '../input/gensim-embeddings-dataset/crawl-300d-2M.gensim',
    '../input/gensim-embeddings-dataset/glove.840B.300d.gensim'
]

NUM_MODELS = 2
# the maximum number of different words to keep in the original texts
# 40_000 is a normal number
# 100_000 seems good too
MAX_FEATURES = 100000

# this is the number of training sample to put in theo model each step
BATCH_SIZE = 512

# units parameters in Keras.layers.LSTM/cuDNNLSTM
# it it the dimension of the output vector of each LSTM cell.
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

EPOCHS = 4

# we will convert each word in a comment_text to a number.
# So a comment_text is a list of number. How many numbers in this list?
# we want the length of this list is a constant -> MAX_LEN
MAX_LEN = 220


def build_matrix(word_index, path):
    # path: a path that contains embedding matrix
    # word_index is a dict of the form ('apple': 123, 'banana': 349, etc)
    embedding_index = KeyedVectors.load(path, mmap='r')
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        for candidate in [word, word.lower()]:
            if candidate in embedding_index:
                embedding_matrix[i] = embedding_index[candidate]
                break
    return embedding_matrix


def build_model(embedding_matrix):
    # a simpler version can be found here
    # https://www.tensorflow.org/tutorials/keras/basic_text_classification

    # Trainable params of the model: 1,671,687
    # Recall that the number of samples in train.csv is 1_804_874

    # words is a vector of MAX_LEN dimension
    words = Input(shape=(MAX_LEN,))

    # Embedding is the keras layer. We use the pre-trained embbeding_matrix
    # https://keras.io/layers/embeddings/
    # have to say that parameters in this layer are not trainable
    # x is a vector of 600 dimension
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)

    # *embedding_matrix.shape is a short way for
    # input_dim = embedding_matrix.shape[0], output_dim  = embedding_matrix.shape[1]

    # here the author used pre-train embedding matrix.
    # instead of train from begining like in tensorflow example

    # https://stackoverflow.com/questions/50393666/how-to-understand-spatialdropout1d-and-when-to-use-it
    x = SpatialDropout1D(0.25)(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)

    hidden = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])

    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='tanh')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, name='main_output')(hidden)

    model = Model(inputs=words, outputs=result)

    # model.summary() will gives a good view of the model structure

    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(clipnorm=0.1),
        metrics=['accuracy', 'mean_squared_error'])

    return model


def main(train_file, test_file):
    print('########')
    print('load ' + train_file.split('/')[-1])
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # Take the columns 'comment_text' from train,
    # then fillall NaN values by emtpy string '' (redundant)
    x_train = train['comment'].fillna('').values

    # if true, y_train[i] =1, if false, it is = 0
    y_train = train['toxicity_score']

    # Take the columns 'comment_text' from test,
    # then fillall NaN values by emtpy string '' (redundant)
    x_test = test['comment'].fillna('').values

    # https://keras.io/preprocessing/text/
    # tokenizer is a class with some method
    tokenizer = text.Tokenizer(num_words=MAX_FEATURES)

    # we apply method fit_on_texts of tokenizer on x_train and x_test
    # it will initialize some parameters/attribute inside tokenizer

    tokenizer.fit_on_texts(list(x_train) + list(x_test))

    # we will convert each word in a comment to a number.
    # So a comment is a list of number.

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)

    # we want the length of this list is a constant -> MAX_LEN
    # if the list is longer, then we cut/trim it
    # if shorter, then we add/pad it with 0's at the beginning
    x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
    x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

    # create an embedding_matrix
    # after this, embedding_matrix is a matrix of size
    # len(tokenizer.word_index)+1   x 600
    # we concatenate two matrices, 600 = 300+300
    embedding_matrix = np.concatenate(
        [build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)
    # embedding_matrix.shape
    # == (410047, 600)

    # embedding_matrix[i] is a 600d vector representation of the word whose index is i
    # embedding_matrix[10]
    # tokenizer.index_word[10] == 'you'

    checkpoint_predictions = []
    weights = []

    print('start training ' + train_file.split('/')[-1])
    # https://keras.io/callbacks/#learningratescheduler

    for model_idx in range(NUM_MODELS):
        # build the same models
        model = build_model(embedding_matrix)
        # We train each model EPOCHS times
        # After each epoch, we reset learning rate (we are using Adam Optimizer)
        # https://towardsdatascience.com/learning-rate-scheduler-d8a55747dd90

        # https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L921
        # learningrate is the attribute 'lr' from Adam optimizer
        # see https://github.com/keras-team/keras/blob/master/keras/optimizers.py#L460
        # In Adam Optimizer, learning rate is changing after each batch
        for global_epoch in range(EPOCHS):
            model.fit(
                x_train,
                y_train,
                batch_size=BATCH_SIZE,
                epochs=1,
                verbose=2,
                callbacks=[
                    LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch), verbose=1)
                ]
            )
            # model.predict will give two outputs: main_output (target) and aux_output
            # we only take main_output
            checkpoint_predictions.append(model.predict(x_test, batch_size=2048).flatten())
            weights.append(2 ** global_epoch)

    # take average (with weights) of predictions from two models
    # predictions is an np.array
    predictions = np.average(np.array(checkpoint_predictions), weights=weights, axis=0)

    test['prediction'] = predictions

    output_file = train_file.split('/')[-1][:-4] + '_predictions.csv'
    test.to_csv(output_file, index=False)
    print('end training ' + train_file.split('/')[-1])


if __name__ == '__main__':
    datasets = ['train_female0.csv', 'train_female0.25.csv', 'train_female0.5.csv', 'train_female0.75.csv',
                'train_female1.csv', ]
    testset = 'test.csv'
    INPUT_PATH = '../input/tox-prediction/data/'
    for dataset in datasets:
        main(INPUT_PATH + dataset, INPUT_PATH + testset)
