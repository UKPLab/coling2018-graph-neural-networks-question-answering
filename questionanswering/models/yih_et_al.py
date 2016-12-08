import keras
from . import QAModel
from . import input_to_indices
import wikidata


class CCNNModel(QAModel):

    def __init__(self, parameters):
        self._main_model, self._sibling_model = self._get_keras_model(parameters)
        self._character2idx = None

    @staticmethod
    def _get_keras_model(p):
        characters_input = keras.layers.Input(shape=(p['max_sent_len'],), dtype='int32', name='sentence_input')
        character_embeddings = keras.layers.Embedding(output_dim=p['emb_dim'], input_dim=p['vocab_size'],
                                                 input_length=p['max_sent_len'],
                                                 mask_zero=False)(characters_input)
        sentence_vector = keras.layers.Convolution1D(p['conv_size'], 3, border_mode='same')(character_embeddings)
        sentence_vector = keras.layers.GlobalMaxPooling1D()(sentence_vector)

        semantic_vector = keras.layers.Dense(p['sem_size'] // 3, activation='tanh')(sentence_vector)
        semantic_vector = keras.layers.Dense(p['sem_size'], activation='tanh', name='semantic_vector')(semantic_vector)
        semantic_vector = keras.layers.Dropout(0.25)(semantic_vector)
        sibiling_model = keras.models.Model(input=[characters_input], output=[semantic_vector])

        sentence_input = keras.layers.Input(shape=(p['max_sent_len'],), dtype='int32', name='sentence_input')
        edge_input = keras.layers.Input(shape=(p['graph_choices'], p['max_sent_len'],), dtype='int32', name='edge_input')

        sentence_vector = sibiling_model(sentence_input)
        edge_vectors = keras.layers.TimeDistributed(sibiling_model)(edge_input)

        main_output = keras.layers.Merge(mode=lambda i: keras.backend.batch_dot(i[0], i[1], axes=(1, 2)), name="edge_scores", output_shape=(p['pool_size'],))([sentence_vector, edge_vectors])
        main_output = keras.layers.Activation('softmax', name='main_output')(main_output)
        model = keras.models.Model(input=[sentence_input, edge_input], output=[main_output])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return model, sibiling_model

    def encode_data(self, data):
        input_set, targets = data
        if not self._character2idx:
            self._character2idx = input_to_indices.get_character_index([" ".join(graphs[0]['tokens']) for graphs in input_set])
        sentences_matrix, edges_matrix = input_to_indices.encode_batch_by_character(data, self._character2idx, wikidata.property2label)
        targets_as_one_hot = keras.utils.np_utils.to_categorical(targets, len(input_set[0]))
        return sentences_matrix, edges_matrix, targets_as_one_hot




