from __future__ import print_function

data_file = '../Data/conv.txt'
clean_data_file = '../Data/conv-clean.txt'

data = None

with open(data_file, mode='r') as f:
    data = f.read().strip().split('\n')

for line_no in range(len(data)):
    try:
        data[line_no] = data[line_no].encode('ascii')
    except:
        data[line_no] = ''

data = filter(lambda s: s != '', data)

with open(clean_data_file, 'wb') as f:
    f.write('\n'.join(data))

U = map(lambda s: s.split('\t'), data)
