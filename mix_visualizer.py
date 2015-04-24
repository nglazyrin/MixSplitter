from scipy.constants.constants import alpha

__author__ = 'Nikolay'


import cPickle
import os
import numpy
import experiment
import tools
import pytz

from matplotlib.pylab import *
from sklearn.preprocessing import normalize
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def draw_spectrum(spectrum, timestamps, num):
    # min and max values for x axis as matplotlib internal dates
    xmin, xmax = mdates.date2num([datetime.datetime.fromtimestamp(timestamps[0]),
                                  datetime.datetime.fromtimestamp(timestamps[-1])])
    print timestamps[0]
    print timestamps[-1]
    print xmin
    print xmax
    fig = matshow(spectrum, fignum=num, aspect='auto', origin='lower', extent=[xmin, xmax, 0, spectrum.shape[0]],
                  cmap='gist_ncar', alpha=1.0) # gist_ncar

    # making the x axis showing time values as HH:MM:SS
    ax = fig.get_axes()
    #ax.xaxis_date()
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M', tz=pytz.timezone('US/Eastern')))  # to start from 00:00:00
    #ax.xaxis.set_major_locator(SecondLocator(interval=300))
    ax.xaxis.set_major_locator(MinuteLocator(byminute=range(0, 85, 5)))
    ax.xaxis.set_minor_locator(SecondLocator(interval=60))
    #fig.get_figure().autofmt_xdate(rotation=0, ha='center')


def draw_selfsim(self_sim, timestamps):
    xmin, xmax = mdates.date2num([datetime.datetime.fromtimestamp(timestamps[0]),
                                  datetime.datetime.fromtimestamp(timestamps[-1])])
    fig = matshow(self_sim, aspect='auto', extent=[xmin, xmax, xmin, xmax], origin='lower')
    ax = fig.get_axes()
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M', tz=pytz.timezone('US/Eastern')))  # to start from 00:00:00
    ax.xaxis.set_major_locator(MinuteLocator(byminute=range(0, 85, 10)))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_major_formatter(DateFormatter('%H:%M', tz=pytz.timezone('US/Eastern')))  # to start from 00:00:00
    ax.yaxis.set_major_locator(MinuteLocator(byminute=range(0, 85, 10)))


def make_average(data, factor, borders):
    for i in range(len(borders) - 1):
        l1 = borders[i]
        r1 = borders[i+1]
        source = numpy.average(data[l1 * factor: r1 * factor], axis=0)
        for j in range(l1 * factor, r1 * factor):
            data[j] = source


def main():
    name = 'Aly & Fila - Future Sound of Egypt 338 '
    print name
    csv_file = os.path.join('.', 'data', name + experiment.plugin_suffix + '.csv')
    cue_file = os.path.join('.', 'data', name + '.cue')
    sim_file = os.path.join('.', 'data', name + experiment.plugin_suffix + '.sim')
    timestamps, data = experiment.read_csv(csv_file)
    info = experiment.read_cue(cue_file)
    has_intro = 'intro' in info[0]['TITLE'].strip().lower()
    has_outro = 'outro' in info[-1]['TITLE'].strip().lower()
    with open(sim_file, 'rb') as f:
        (self_sim, factor) = cPickle.load(f)

    filtered, novelty = tools.init_borders2(self_sim)
    sec_per_row = timestamps[-1] / self_sim.shape[0]
    #filtered = [x * sec_per_row for x in filtered]
    filtered = tools.detect_track_borders(data,
                                          timestamps[-1],
                                          len(info),
                                          self_sim=self_sim,
                                          factor=factor,
                                          has_intro=has_intro,
                                          has_outro=has_outro)
    #make_average(data, factor, [int(x / sec_per_row) for x in filtered])
    #for i in range(len(info)):
    #    print '%d.\t%.2f\t%.2f: %f' % (i, info[i]['INDEX'], filtered[i], filtered[i] - info[i]['INDEX'])
    #print filtered
    #print [x['INDEX'] for x in info]
    #print experiment.get_diff(info, filtered)

    data = data.transpose((1, 0))
    draw_spectrum(data, timestamps, 1)
    #draw_selfsim(self_sim, timestamps)
    vlines([datetime.datetime.fromtimestamp(x) for x in filtered], 0, 24, color='k', linewidths=[1]*len(filtered), alpha=1.0)
    vlines([datetime.datetime.fromtimestamp(x['INDEX']) for x in info], 24, 48, color='y', linewidths=[1]*len(info), alpha=1.0)
    #savefig('fig.pdf', bbox_inches='tight')

    tracks = [(x['PERFORMER'], x['TITLE']) for x in info]
    export_cue(name, tracks, filtered)

    # borders2, novelty = tools.init_borders2(self_sim)
    # plt.figure(2)
    # plt.plot(novelty)
    # print borders2

    #draw_spectrum(data2, timestamps, 2)
#    plt.figure(2)
#    plt.plot(data2)

#    plt.figure(3)
#    plt.matshow(areas, fignum=3)
#    _, axs = plt.subplots(5, 1, sharex=True)
#    tss = [datetime.datetime.fromtimestamp(x) for x in timestamps]
#    for i in range(len(centroids)):
#        axs[i].plot(centroids[i])
#    axs[4].plot(flux)

    # plt.figure(4)
    # plt.matshow(self_sim, fignum=4)

    show()


def export_cue(filename, tracks, timestamps):
    cue_file = os.path.join('.', 'data', filename + '-splitted.cue')
    performer, title = get_performer_and_title(filename)
    with open(cue_file, 'wb') as output:
        print >>output, 'PERFORMER "%s"' % performer
        print >>output, 'TITLE "%s"' % title
        print >>output, 'FILE "%s.mp3" MP3' % filename
        indent = '  '
        for i in range(len(tracks)):
            total = timestamps[i]
            minutes = int(total) / 60
            seconds = int(total) % 60
            cents = int(total * 100) % 100
            print >>output, '%sTRACK %02d AUDIO' % (indent, i+1)
            print >>output, '%sPERFORMER "%s"' % (2*indent, tracks[i][0])
            print >>output, '%sTITLE "%s"' % (2*indent, tracks[i][1])
            print >>output, '%sINDEX 01 %02d:%02d:%02d' % (2*indent, minutes, seconds, cents)


def get_performer_and_title(filename):
    separator = None
    if ' - ' in filename:
        separator = ' - '
    elif '_-_' in filename:
        separator = '_-_'
    elif '-' in filename:
        separator = '-'
    elif '_' in filename:
        separator = '_'
    else:
        separator = ' '
    segments = filename.split(separator)
    if len(segments) <= 1:
        return filename, filename
    if len(segments) == 2:
        return segments[0], segments[1]
    elif segments[0].isdigit():
        return segments[1], segments[2:]
    else:
        return segments[0], segments[1:]


if __name__ == '__main__':
    main()
