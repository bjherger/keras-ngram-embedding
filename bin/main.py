#!/usr/bin/env python
"""
coding=utf-8

Code template courtesy https://github.com/bjherger/Python-starter-repo

"""
import datetime
import logging
import os
import pickle

import numpy
from keras import Input, Model
from keras.callbacks import TensorBoard
from keras.layers import Embedding, Dense, Flatten
from keras.preprocessing.sequence import skipgrams

from bin import transformations


def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.basicConfig(level=logging.DEBUG)

    # Read in text
    text_path = '../data/alice_in_wonderland.txt'
    text = open(text_path, 'r').read()
    logging.info('Read {} characters from {}'.format(len(text), text_path))

    # Change text to sequence of indices
    vectorizer = transformations.EmbeddingVectorizer()
    indices = vectorizer.fit_transform([[text]])

    vocab_size = numpy.max(indices) + 1

    # Change sequence of indices to skipgram training pair and T/F label (E.g. [[project, gutenberg], True]
    # TODO There must be a better way of getting a 1d array
    X, y = skipgrams(indices.tolist()[0], vocabulary_size=vocab_size, window_size=4, categorical=True)
    X = numpy.array(X)
    y = numpy.array(y)
    logging.info('X shape: {}, y shape: {}'.format(X.shape, y.shape))


    # Create architecture
    # TODO Should be two separate inputs, rather than a timeseries w/ 2 time steps
    input_layer = Input(shape=(2,), name='text_input')
    x = input_layer
    x = Embedding(input_dim=vocab_size, output_dim=50, input_length=2, name='text_embedding')(x)
    x = Flatten()(x)
    x = Dense(2, activation='softmax', name='output')(x)

    model = Model(input_layer, x)

    model.compile(optimizer='Adam', loss='categorical_crossentropy')

    # Train architecture
    callbacks = [TensorBoard(os.path.expanduser('~/.logs/'+str(datetime.datetime.now())))]
    model.fit(X, y, epochs=5, validation_split=.1, callbacks=callbacks, batch_size=2**13)

    embedding = model.get_layer('text_embedding')
    weights = embedding.get_weights()[0]
    print(weights)
    print(weights.shape)
    print(type(weights))

    # Store weights
    pickle.dump(weights, open('alice_embedding.pkl', 'wb'))
    pickle.dump(vectorizer.token_index_lookup, open('alice_vocab_index.pkl', 'wb'))


# Main section
if __name__ == '__main__':
    main()
