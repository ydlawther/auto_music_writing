import numpy as np
from notes import *
from model import *


def train():
    notes=get_notes()
    notes_len=len(set(notes))
    note_name=sorted(set(i for i in notes))  # 获得排序的不重复的音符名字
    sequence_length = 11  # 序列长度     这里修改过 100个改为10个
    note_dict=dict((j,i) for i,j in enumerate(note_name))  # 设计一个字典，把音符转换成数字，方便训练
    network_input = []  # 创建输入序列
    network_output = []  # 创建输出序列

    # 创建输入和输出序列
    for i in range(0, len(notes) - sequence_length):
        # 输入100个，输出1个
        sequence_in = notes[i: i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_dict[k] for k in sequence_in])
        network_output.append(note_dict[sequence_out])

    # 将输入序列转化为符合LSTM层的形式
    network_input = np.reshape(network_input, (len(network_input), sequence_length, 1))
    #归一化
    normal_network_input = network_input / float(notes_len)

    network_output = tf.keras.utils.to_categorical(network_output)  # 输出布尔矩阵，配合categorical_crossentropy 算法使用
    model =get_model(normal_network_input,notes_len)
    file_path = "weights-{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        file_path,
        monitor='loss',  # 监控的对象是loss
        verbose=0,
        save_best_only=True,
        mode='min'  # 如果监控对象是val_acc则取max，是loss则取min
    )
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=1, batch_size=128, callbacks=callbacks_list)  # 整体迭代1次，每小批128个   这里修改过，100次改为5次
    return network_input, normal_network_input, notes_len, note_name


if __name__ == '__main__':
    # train()#训练的时候执行
    train()
