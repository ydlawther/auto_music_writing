import subprocess
import os


def MIDItoMP3(input_file):
    for i in range(1, 100):
        if os.path.exists('output%d.mp3' % i):
            i += 1
        else:
            break
    output_file = 'output%d.mp3' % i
    assert os.path.exists(input_file)
    print('Conveting %s to MP3' % input_file)
    command = 'timidity {} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 256k {}'.format(input_file, output_file)
    subprocess.call(command, shell=True)
    print('Converted. Generated file is %s' % output_file)

# if __name__ == '__main__':
# MIDItoMP3('file_path')
