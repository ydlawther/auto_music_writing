# coding=utf-8
import os
from music21 import converter, instrument, note, chord


# 读取训练数据的Notes
def get_notes():
    filepath = 'data_sets/'
    files = os.listdir(filepath)
    nOtes = []

    for file in files:
        try:
            stream = converter.parse(filepath + file)
            instru = instrument.partitionByInstrument(stream)
            if instru:  # 如果有乐器部分，取第一个乐器部分
                notes = instru.parts[0].recurse()
            else:  # 如果没有乐器部分，直接取note
                notes = stream.flat.notes

            for element in notes:
                # 如果是 Note 类型，取音调
                # 如果是 Chord 类型，取音调的序号,存int类型比较容易处理
                if isinstance(element, note.Note):
                    nOtes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    nOtes.append('.'.join(str(n) for n in element.normalOrder))
        except:
            pass
        # with open('Note %d'% i, 'a+')as f:
             # f.write(str(Notes))
    return nOtes

# if __name__ == '__main__':
    #train()#训练的时候执行
    # get_notes()'''
