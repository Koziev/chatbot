import io

input_file = '../../data/sample_text.txt'
output_file = '../../tmp/dialogues.txt'

with io.open(input_file, 'r', encoding='utf-8') as rdr,\
    io.open(output_file, 'w', encoding='utf-8') as wrt:
    prev_line = ''
    for line in rdr:
        line = line.strip()
        if line:
            if line.startswith('—'):
                if not prev_line.startswith('—'):
                    wrt.write('\n')

                wrt.write('{}\n'.format(line))

            prev_line = line

