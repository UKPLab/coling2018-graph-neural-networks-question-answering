from keras import layers, models, callbacks


def SiameseCNN(p, max_sent_len, max_property_len, embeddings):
    sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
    word_embeddings = layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                       input_length=max_sent_len, weights=[embeddings],
                                       mask_zero=False, trainable=False)(sentence_input)
    edge_input = layers.Input(shape=(max_property_len,), dtype='int32', name='edge_input')
    edge_embeddings = layers.Embedding(output_dim=embeddings.shape[1], input_dim=embeddings.shape[0],
                                       input_length=max_property_len,  weights=[embeddings],
                                       mask_zero=False, trainable=False)(edge_input)

    #     sentence_vector = layers.LSTM(100, consume_less='gpu')(word_embeddings)
    #     edge_vector = layers.LSTM(100, consume_less='gpu')
    sentence_vector = layers.GlobalMaxPooling1D()(word_embeddings)
    edge_vector = layers.GlobalMaxPooling1D()(edge_embeddings)
    x = layers.merge([sentence_vector, edge_vector], mode="concat")
    x = layers.Dense(50, activation='tanh')(x)
    main_output = layers.Dense(1, activation='sigmoid', name='main_output')(x)

    model = models.Model(input=[sentence_input, edge_input], output=[main_output])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model
