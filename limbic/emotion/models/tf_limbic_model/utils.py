import json
import os
import pickle
from datetime import datetime
from typing import Iterable, List, NamedTuple, Optional

import numpy as np
import tensorflow as tf

_FILE_PATH = os.path.dirname(__file__)
_MODELS_PATH = os.path.join(_FILE_PATH, '../../../models')


class ModelParams(NamedTuple):
    max_len: int
    max_words: int
    embedding_size: int
    drop_out_rate: float
    labels: List[str]
    loss_function: str
    adam_lr_parameter: float
    model_metrics: List[str]


def load_book(lines: Iterable[str]) -> List[str]:
    """
    Each paragraph will be determined by the fact there's an empty line.
    i.e. this will add lines to every paragraph until we get a new empty line.
    """
    paragraphs = []
    potential_paragraph: List[str] = []
    for line in lines:
        if line.strip() == '':
            paragraphs.append(' '.join(potential_paragraph).strip())
            potential_paragraph = []
        potential_paragraph.append(line.strip())
    return paragraphs


def load_glove_embeddings(glove_file: str):  # TODO: add the actual types
    """
    This should only work for Glove embeddings for now, as it's specifically tailored
    to load such file format. Each Glove array will be returned as dtype float32 to optimize for memory space.
    """
    embeddings_index = {}
    with open(glove_file, encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            embed = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embed
    return embeddings_index


def preprocess_sentence(sentence: str) -> str:
    """
    The idea is to have this as clean as possible to match the embeddings.

    In the meantime I'll just keep alphabetic characters amd remove all white space chars.
    """
    return ''.join([c for c in ' '.join(sentence.split())
                    if c.isalpha() or c == ' ']).lower().strip()


def process_train(train, categories: List[str], max_words: int, max_len: int):
    """
    This first gets all sentences from the dataframe, cleans them, and then
    fits a tokenizer object with all the sentences. Note that it only keeps a maximum
    number of words so it doesn't blow up (based on top frequent words in the corpus).
    Finally, creates the final training object transforming the sentence into a
    numpy array with the token numbers (associated to the tokeinzer) and adding the padding.
    """
    x_train = train['text'].str.lower().apply(lambda x: preprocess_sentence(x))
    y_train = train[categories].values

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, lower=True)
    tokenizer.fit_on_texts(x_train)

    x_train = tokenizer.texts_to_sequences(x_train)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    return x_train, y_train, tokenizer


def build_embeddings_matrix(tokenizer, embeddings_file: str, max_words: int, embedding_size: int):
    """
    This should return a matrix with MAX_WORDS x EMBEDDING_SIZE to be used
    in the input layer of the neural network.

    Note that if a word is not in the embeddings, this will return a Zeros array.
    """
    embeddings_index = load_glove_embeddings(embeddings_file)
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((max_words, embedding_size), dtype='float32')

    for word, i in word_index.items():
        if i > max_words:
            break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def build_model(embedding_matrix, params: ModelParams):
    """
    This method will piece together the final neural network that will be used
    to train the emotions for every sentence, which is within the family of multi-label classification models,
    where more than one label is accepted to be part of the prediction output and
    it's allowed a sentence to have many emotions in "equal quantities".

    The general idea is the following:

    (If you are reading through this and you think I should've done something different, let me know!)

    1. use the embeddings layer first leveraging a pre-trained layer (e.g. Glove in this case)
    2. create a bidirectional layer to capture contextual info from upstream and downstream directions
    3. allow some drop-out to include some regularization into the model and avoid over-fitting
    4. use a convolution layer to extract some relationships between the hypothesis from the previous layer
    5. using a pooling layer (avg and max) to group previous convolutions
    6. use sigmoid activation function in the output layer for the multi-label problem
    7. use binary cross-entropy loss function which is well suited for the multi-label problem.

    More details and some references included in each step below.
    """

    # TODO: Configure more parameters, e.g. amount of units in intermediate layers, activation functions per step, etc.
    max_len = params.max_len
    max_words = params.max_words
    embedding_size = params.embedding_size
    drop_out_rate = params.drop_out_rate
    labels = params.labels
    loss_function = params.loss_function
    adam_lr_parameter = params.adam_lr_parameter
    model_metrics = params.model_metrics

    input_layer = tf.keras.layers.Input(shape=(max_len, ))

    # First we setup the input layer with out pre-trained embeddings (non trainable as I'm assuming Glove is enough)
    x = tf.keras.layers.Embedding(max_words,
                                  embedding_size,
                                  weights=[embedding_matrix],
                                  trainable=False)(input_layer)

    # Bi-directional LSTM will identify some patterns from both directions of a sentence
    # Also, will use GRU instead of other recurrent gates (e.g. LSTM) as per https://arxiv.org/pdf/1702.01923.pdf
    # Drop-out also allows to have some regularization https://www.aclweb.org/anthology/D14-1181.pdf
    # more info: https://mlwhiz.com/blog/2018/12/17/text_classification/
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(256,
                            return_sequences=True,
                            dropout=drop_out_rate,
                            recurrent_dropout=drop_out_rate))(x)

    # then use a convolutional layer to capture finer relationships between co-allocated words.
    # using kernel size 3 to group in windows of 5 words
    # exploring Lecun uniform initializer more info: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
    # more on this: https://medium.com/usf-msds/deep-learning-best-practices-1-weight-initialization-14e5c0295b94
    x = tf.keras.layers.Conv1D(128,
                               activation='relu',
                               kernel_size=5,
                               kernel_initializer="lecun_uniform")(x)

    # TODO: explore using 2D max pooling as it seems to be a good alternative for sentiment classification
    # check: https://www.aclweb.org/anthology/C16-1329.pdf
    #     x = tf.keras.layers.GlobalMaxPooling1D()(x)

    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)

    x = tf.keras.layers.concatenate([avg_pool, max_pool])

    # predictions using sigmoid activation function to approximate probabilities for each label
    # note that we should not use softmax as this is not a multi-category problem, but a multi-label problem.
    prediction_layer = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)
    model = tf.keras.Model(input_layer, prediction_layer)

    # Given that this is a multi-label problem with a sigmoid activation function,
    # using a binary cross-entropy is best suited, check: https://gombru.github.io/2018/05/23/cross_entropy_loss/
    # using Adam well known for it's efficiency and performance vs other optimizers.
    model.compile(loss=loss_function,
                  optimizer=tf.keras.optimizers.Adam(lr=adam_lr_parameter),
                  metrics=model_metrics)
    return model


def save_model(tokenizer, model, training_metadata):
    """
    Save the Tokenizer and the Model to disk for later usage.

    Also keep some metadata on the parameters and data sources used to train the model.

    For version control, I'll use the current date as there's no need to more granularity.
    """
    current_date = datetime.now().date().isoformat()
    metadata_file_path = f'model_metadata_{current_date}.txt'
    tokenizer_file_path = f'tokenizer_{current_date}.pickle'
    model_file = f'emotions_model_{current_date}.h5'

    training_metadata['model_version_date'] = current_date
    training_metadata['metadata_file'] = metadata_file_path
    training_metadata['tokenizer_file'] = tokenizer_file_path
    training_metadata['model_file'] = model_file

    with open(metadata_file_path, 'w') as meta:
        meta.write(json.dumps(training_metadata, indent=2))

    with open(tokenizer_file_path, 'wb') as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)

    model.save(model_file)


def load_model(experiments_date: str):
    """
    Load the Tokenizer and the Model from disk.
    """
    model_path = os.path.join(_MODELS_PATH, f'emotions_model_{experiments_date}.h5')
    tokenizer_path = os.path.join(_MODELS_PATH, f'tokenizer_{experiments_date}.pickle')

    with open(tokenizer_path, 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    model = tf.keras.models.load_model(model_path)

    return model, tokenizer


def continuous_labels_to_binary(labels, threshold: Optional[float] = 0.25):
    label_y_test = []
    for y in labels:
        integer_labels = []
        for i in y:
            if i > threshold:
                integer_labels.append(1)
            else:
                integer_labels.append(0)
        label_y_test.append(integer_labels)
    return np.array(label_y_test)
