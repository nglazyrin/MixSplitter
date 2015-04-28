__author__ = 'Nikolay'

import numpy
import scipy.stats
import scipy.stats.mstats
import os
from experiment import read_cue, get_diff


def build_even(length, tracks, has_intro=False, has_outro=False, intro_size=0, outro_size=0):
    left = 0
    right = length
    result = [0]
    if has_intro:
        result.append(intro_size)
        left = intro_size
        tracks -= 1
    if has_outro:
        right = length - outro_size
        tracks -= 1
    result.extend([left + i * (right - left) * 1.0 / tracks for i in range(1, tracks)])
    if has_outro:
        result.append(length - outro_size)
    result.append(length)
    return result


# print build_even(10, 10)
# print build_even(10, 8, True, False, 3, 5)
# print build_even(10, 8, False, True, 5, 3)
# print build_even(10, 6, True, True, 3, 3)

def fill_frame_pairs(array):
    track_i = 0
    track_j = 0
    pairs = set()
    for i in range(int(length)):
        if i >= array[track_i + 1]:
            track_i += 1
        j = i
        if j >= array[track_j + 1]:
            track_j += 1
        while j < array[track_j + 1] and track_i == track_j:
            pairs.add((i, j))
            j += 1
    return pairs


def as_time(seconds):
    m, s = divmod(seconds, 60)
    return "%02d:%02d" % (m, s)


def intersect(borders, true_borders, delta=120):
    added = []
    true_matched = []
    for border in borders:
        matched = False
        for true_border in true_borders:
            if abs(true_border - border) < delta and not true_border in true_matched:
                true_matched.append(true_border)
                matched = True
                break
        if not matched:
            added.append(border)
    removed = list(set(true_borders).difference(set(true_matched)))
    added = [as_time(s) for s in added]
    removed = [as_time(s) for s in sorted(removed)]
    return added, removed

avg_sum = 0
max_sum = 0
even_avg_sum = 0
even_max_sum = 0
count = 0
good = 0
even_good = 0
avg_precision = 0
avg_recall = 0
avg_f_measure = 0
mix_styles = {}
with open('mix_list.txt', 'rb') as mixlist:
    for line in mixlist:
        rec = line.strip().split('\t')
        mix_styles[rec[0]] = rec[1]

proposed = []
naive = []
style_stats = {}
intro_size = 30
outro_size = 30
with open(os.path.join('logs', 'test_beat_profile.log'), 'rb') as log:
    for line in log:
        rec = line.strip().split('\t')
        if len(rec) > 2:
            name = rec[0]
            info = read_cue(os.path.join('temp_data', name[:-4] + '.cue'))
            # indices = numpy.array([info[i+1]['INDEX'] - info[i]['INDEX'] for i in range(len(info)-1)])
            indices = [info[i]['INDEX'] for i in range(len(info))]
            has_intro = 'intro' in info[0]['TITLE'].lower()
            has_outro = 'outro' in info[-1]['TITLE'].strip().lower()
            # has_intro = False
            # has_outro = False
            avg = float(rec[1])
            max_diff = float(rec[2])

            borders = rec[3][1:-1].split(',')
            borders = [float(x) for x in borders]
            length = borders[-1]
            indices.append(length)
            tracks = len(info)
            even = build_even(length, tracks, has_intro, has_outro, intro_size, outro_size)

            even_avg, even_max = get_diff(info, even)
            even_avg_sum += even_avg
            even_max_sum += even_max

            P_E = fill_frame_pairs(borders)
            # P_E = fill_frame_pairs(even)
            P_A = fill_frame_pairs(indices)
            precision = len(P_E.intersection(P_A)) * 1.0 / len(P_E)
            recall = len(P_E.intersection(P_A)) * 1.0 / len(P_A)
            f_measure = 2 * precision * recall / (precision + recall)
            avg_precision += precision
            avg_recall += recall
            avg_f_measure += f_measure

            print '%s\t%f\t%f\t%d\t%f\t%f\t%f' % (name, avg, even_avg, len(info), precision, recall, f_measure)
            proposed.append(avg)
            naive.append(even_avg)

            added, removed = intersect(borders, indices)
            if max_diff > 60:
                print '\tAdded: %s' % added
                print '\tRemoved: %s' % removed

            maximum = float(rec[2])
            # if avg < 0.5 * length/tracks:
            if avg < 30 and max_diff < 60:
                good += 1
            # if even_avg < 0.5 * length/tracks:
            if even_avg < 30 and even_max < 60:
                even_good += 1
            avg_sum += avg
            max_sum += maximum
            count += 1

            '''
            if mix_styles[name] not in style_stats:
                style_stats[mix_styles[name]] = {'count': 0, 'sum_avg': 0, 'sum_max': 0, 'sum_even_avg': 0, 'sum_even_max': 0}
            style_stats[mix_styles[name]]['count'] += 1
            style_stats[mix_styles[name]]['sum_avg'] += avg
            style_stats[mix_styles[name]]['sum_max'] += maximum
            style_stats[mix_styles[name]]['sum_even_avg'] += even_avg
            style_stats[mix_styles[name]]['sum_even_max'] += even_max
            '''

print '%f\t%f' % (avg_sum * 1.0 / count, even_avg_sum * 1.0 / count)
print '%f\t%f' % (max_sum * 1.0 / count, even_max_sum * 1.0 / count)
print '%f\t%f\t%f' % (avg_precision * 1.0 / count, avg_recall * 1.0 / count, avg_f_measure * 1.0 / count)
print '%d\t%d\n\n' % (good, even_good)


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
