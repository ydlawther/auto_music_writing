import shutil
from train import *
from convert import MIDItoMP3
from music21 import stream


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 忽略警告：Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA


def generate_notes(model, network_input, note_name, notes_len):
    randindex = np.random.randint(0, len(network_input) - 1)
    notedic = dict((i,j) for i, j in enumerate(note_name))    # 把刚才的整数还原成音调
    pattern = network_input[randindex]
    prediction = []
    #随机生成1000个音符
    for note_index in range(1000):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(notes_len)#归一化
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = notedic[index]
        prediction = np.append(prediction, result)
        # 往后移动
        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]
    return prediction


# 生成mid音乐
def create_music():
    network_input, normal_network_input, notes_len, note_name = train()
    # 寻找loss最小的weight文件，作为训练参数
    files = os.listdir()
    min_loss = {}
    for i in files:
        if 'weights' in i:
            num = i[11:15]
            min_loss[num] = i
    best_weights = min_loss[min(min_loss.keys())]
    model = get_model(normal_network_input, notes_len,best_weights)
    prediction = generate_notes(model, network_input, note_name, notes_len)
    offset = 0
    output_notes = []
    # 生成 Note（音符）或 Chord（和弦）对象
    for data in prediction:
        if ('.' in data) or data.isdigit():
            notes_in_chord = data.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes = np.append(notes,new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes = np.append(output_notes, new_chord)
        else:
            new_note = note.Note(data)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes = np.append(output_notes, new_note)
        offset += 0.5
    # 创建音乐流（Stream）
    midi_stream = stream.Stream(output_notes)
    # 写入 MIDI 文件
    midi_stream.write('midi', fp='output.mid')


if __name__ == '__main__':
    # train()#训练的时候执行
    create_music()
    for i in range(1,100):
        if os.path.exists('output%d.mid' % i):
            i += 1
        else:
            shutil.move("output.mid", "output%d.mid" % i)
            MIDItoMP3("output%d.mid" % i)
            break
