__author__ = 'Nikolay'

import numpy
import os
from experiment import read_cue, get_diff

avg_sum = 0
max_sum = 0
even_avg_sum = 0
even_max_sum = 0
count = 0
good = 0
even_good = 0
with open(os.path.join('logs', 'paper_test_explicit_intro_outro.log'), 'rb') as log:
    for line in log:
        rec = line.strip().split('\t')
        if len(rec) > 2:
            name = rec[0]
            info = read_cue(os.path.join('data', name[:-4] + '.cue'))
            indexes = numpy.array([info[i+1]['INDEX'] - info[i]['INDEX'] for i in range(len(info)-1)])
            has_intro = 'intro' in info[0]['TITLE'].lower()
            has_outro = 'outro' in info[-1]['TITLE'].strip().lower()
            #has_intro = False
            #has_outro = False
            avg = float(rec[1])

            borders = rec[3][1:-1].split(',')
            length = float(borders[-1])
            tracks = len(info)
            left = 0
            right = length
            if has_intro:
                intro_size = 30
                even = [0]
                left = intro_size
                tracks -= 1
                even.extend([intro_size + i * (length - intro_size) / (tracks - 1) for i in range(tracks-1)])
            else:
                even = [i * length / tracks for i in range(tracks)]
            if has_outro:
                outro_size = 30
                right = length - outro_size
                tracks -= 1
            even = [i * (right - left) / tracks for i in range(tracks)]
            if has_intro:
                temp = [0]
                temp.extend([x + intro_size for x in even])
                even = temp
            if has_outro:
                even.append(length)

            even_avg, even_max = get_diff(info, even)
            even_avg_sum += even_avg
            even_max_sum += even_max

            print '%s\t%f\t%f\t%d' % (name, avg, even_avg, len(info))

            maximum = float(rec[2])
            #if avg < 0.5 * length/tracks:
            if avg < 30:
                good += 1
            #if even_avg < 0.5 * length/tracks:
            if even_avg < 30:
                even_good += 1
            avg_sum += avg
            max_sum += maximum
            count += 1
print '%f\t%f' % (avg_sum / count, even_avg_sum / count)
print '%f\t%f' % (max_sum / count, even_max_sum / count)
print '%d\t%d'% (good, even_good)