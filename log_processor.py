__author__ = 'Nikolay'

import numpy
import scipy.stats
import scipy.stats.mstats
import os
from experiment import read_cue, get_diff

avg_sum = 0
max_sum = 0
even_avg_sum = 0
even_max_sum = 0
count = 0
good = 0
even_good = 0
mix_styles = {}
with open('mix_list.txt', 'rb') as mixlist:
    for line in mixlist:
        rec = line.strip().split('\t')
        mix_styles[rec[0]] = rec[1]

proposed = []
naive = []
style_stats = {}
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
            proposed.append(avg)
            naive.append(even_avg)

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

            if mix_styles[name] not in style_stats:
                style_stats[mix_styles[name]] = { 'count': 0, 'sum_avg': 0, 'sum_max': 0 , 'sum_even_avg': 0, 'sum_even_max': 0}
            style_stats[mix_styles[name]]['count'] += 1
            style_stats[mix_styles[name]]['sum_avg'] += avg
            style_stats[mix_styles[name]]['sum_max'] += maximum
            style_stats[mix_styles[name]]['sum_even_avg'] += even_avg
            style_stats[mix_styles[name]]['sum_even_max'] += even_max
print '%f\t%f' % (avg_sum / count, even_avg_sum / count)
print '%f\t%f' % (max_sum / count, even_max_sum / count)
print '%d\t%d\n\n'% (good, even_good)


print scipy.stats.normaltest(proposed)
print scipy.stats.normaltest(naive)
print scipy.stats.wilcoxon(proposed, naive)

for style, stats in sorted(style_stats.items(), key=lambda x: x[1]['count'], reverse=True):
    stats['sum_avg'] /= stats['count']
    stats['sum_max'] /= stats['count']
    stats['sum_even_avg'] /= stats['count']
    stats['sum_even_max'] /= stats['count']
    print '  %s & %s & %d & %.2f s & %.2f s \\\\ ' % (style, 'Proposed', stats['count'], stats['sum_avg'], stats['sum_max'])
    print '  %s & %s & %d & %.2f s & %.2f s \\\\ ' % (style, 'Naive', stats['count'], stats['sum_even_avg'], stats['sum_even_max'])
    print '  \\hline'
