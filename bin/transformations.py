import logging
from collections import defaultdict

from gensim.utils import simple_preprocess
import numpy
from sklearn.base import TransformerMixin, BaseEstimator


class EmbeddingVectorizer(TransformerMixin, BaseEstimator):
    """
    Converts text into padded sequences. The output of this transformation is consistent with the required format
    for Keras embedding layers
    For example `'the fat man`' might be transformed into `[2, 1, 27, 2, 2, 2]`, if the `embedding_sequence_length` is
    6.
    There are a few sentinel values used by this layer:
     - `0` is used for the UNK token (tokens which were not seen during training)
     - `1` is used for the padding token (to fill out sequences that shorter than `embedding_sequence_length`)
    """

    def __init__(self, max_sequence_length=None):
        # TODO Allow for UNK 'dropout' rate

        self.max_sequence_length = max_sequence_length

        # Create a dictionary, with default value 0 (corresponding to UNK token)
        self.token_index_lookup = defaultdict(int)
        self.token_index_lookup['UNK'] = 1
        self.token_index_lookup['__PAD__'] = 2
        self.next_token_index = 3

        pass

    def fit(self, X, y=None):
        # Format text for processing, by creating a list of strings
        observations = self.prepare_input(X)

        # Preprocess & tokenize
        observations = list(map(lambda x: simple_preprocess(x), observations))

        # Generate embedding_sequence_length, if necessary
        if self.max_sequence_length is None:
            self.max_sequence_length = self.generate_embedding_sequence_length(observations)

        # Update index_lookup
        tokens = set()
        for observation in observations:
            tokens.update(observation)

        logging.debug('Fitting with tokens: {}'.format(tokens))

        current_max_index = max(self.token_index_lookup.values())
        index_range = range(current_max_index, len(tokens) + current_max_index)
        learned_token_index_lookup = dict(zip(tokens, index_range))
        self.token_index_lookup.update(learned_token_index_lookup)
        new_max_token_index = max(self.token_index_lookup.values())
        logging.info('Learned tokens, new_max_token_index: {}'.format(new_max_token_index))
        return self

    def transform(self, X):
        observations = self.prepare_input(X)

        # Convert to embedding format
        observations = list(map(self.process_string, observations))

        # Redo numpy formatting
        observations = list(map(lambda x: numpy.array(x), observations))

        X = numpy.matrix(observations)
        logging.info('Transformed text, max index: {}'.format(numpy.max(X)))

        return X

    @staticmethod
    def generate_embedding_sequence_length(observation_series):
        lengths = list(map(len, observation_series))
        embedding_sequence_length = max([int(numpy.median(lengths)), 1])
        logging.info('Generated embedding_sequence_length: {}'.format(embedding_sequence_length))

        return embedding_sequence_length

    def process_string(self, input_string):
        """
        Turn a string into padded sequences, consistent with Keras's Embedding layer
         - Simple preprocess & tokenize
         - Convert tokens to indices
         - Pad sequence to be the correct length
        :param input_string: A string, to be converted into a padded sequence of token indices
        :type input_string: str
        :return: A padded, fixed-length array of token indices
        :rtype: [int]
        """
        logging.debug('Processing string: {}'.format(input_string))

        # Convert to tokens
        tokens = simple_preprocess(input_string)
        logging.debug('Tokens: {}'.format(tokens))

        # Convert to indices
        indices = list(map(lambda x: self.token_index_lookup[x], tokens))
        logging.debug('Indices: {}'.format(indices))

        # Pad indices
        padding_index = numpy.nan
        padding_length = self.max_sequence_length
        padded_indices = self.pad(indices, length=padding_length, pad_char=padding_index)
        logging.debug('Padded indices: {}'.format(padded_indices))

        return padded_indices

    @staticmethod
    def pad(input_sequence, length, pad_char):
        """
        Pad the given iterable, so that it is the correct length.
        :param input_sequence: Any iterable object
        :param length: The desired length of the output.
        :type length: int
        :param pad_char: The character or int to be added to short sequences
        :type pad_char: str or int
        :return: A sequence, of len `length`
        :rtype: []
        """

        # If input_sequence is a string, convert to to an explicit list
        if isinstance(input_sequence, str):
            input_sequence = list(input_sequence)

        # If the input_sequence is the correct length, return it
        if len(input_sequence) == length:
            return input_sequence

        # If the input_sequence is too long, truncate it
        elif len(input_sequence) > length:
            return input_sequence[:length]

        # If the input_sequence is too short, extend it w/ the pad_car
        else:
            padding_len = length - len(input_sequence)
            padding = [pad_char] * padding_len
            return list(input_sequence) + list(padding)

    @staticmethod
    def prepare_input(X):
        # Undo Numpy formatting
        observations = list(map(lambda x: x[0], X))

        observations = map(str, observations)
        return observations
