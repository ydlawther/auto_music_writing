import tensorflow as tf


def get_model(inputs, notes_len, weights_file=None):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(512, input_shape=(inputs.shape[1], inputs.shape[2]),
                                   return_sequences=True))  # 512层神经元，return_sequences=True表示返回所有的输出序列
    model.add(tf.keras.layers.Dropout(0.3))  # 丢弃 30% 神经元，防止过拟合
    model.add(tf.keras.layers.LSTM(512, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.LSTM(512))  # return_sequences 是默认的 False，只返回输出序列的最后一个
    model.add(tf.keras.layers.Dense(256))  # 256 个神经元的全连接层
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(notes_len))  # 输出的数目等于所有不重复的音调的数目
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    if weights_file is not None:
        model.load_weights(weights_file)

    return model
