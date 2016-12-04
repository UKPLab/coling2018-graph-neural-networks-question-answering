from keras import layers, models, callbacks


def SiameseCNN(p):
    tokens_input = layers.Input(shape=(p['max_sent_len'],), dtype='int32', name='sentence_input')
    word_embeddings = layers.Embedding(output_dim=p['emb_dim'], input_dim=p['vocab_size'],
                                       input_length=p['max_sent_len'],
                                       mask_zero=False)(tokens_input)
    sentence_vector = layers.Convolution1D(p['conv_size'], 3, border_mode='same')(word_embeddings)
    sentence_vector = layers.GlobalMaxPooling1D()(sentence_vector)
    semantic_vector = layers.Dense(p['sem_size'], activation='tanh', name='semantic_vector')(sentence_vector)
    sibiling_model = models.Model(input=[tokens_input], output=[semantic_vector])

    sentence_input = layers.Input(shape=(p['max_sent_len'],), dtype='int32', name='sentence_input')
    edge_input = layers.Input(shape=(p['max_sent_len'],), dtype='int32', name='edge_input')

    sentence_vector = sibiling_model(sentence_input)
    edge_vector = sibiling_model(edge_input)
    main_output = layers.merge([sentence_vector, edge_vector], mode="cos", name='main_output')

    model = models.Model(input=[sentence_input, edge_input], output=[main_output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
