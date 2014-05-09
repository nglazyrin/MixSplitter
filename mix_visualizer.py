__author__ = 'Nikolay'


import cPickle
import os
import experiment
import tools

from matplotlib.pylab import *
from sklearn.preprocessing import normalize
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def draw_spectrum(spectrum, timestamps, num):
    # min and max values for x axis as matplotlib internal dates
    xmin, xmax = mdates.date2num([datetime.datetime.fromtimestamp(timestamps[0]), datetime.datetime.fromtimestamp(timestamps[-1])])
    fig = matshow(spectrum, fignum=num, aspect='auto', origin='lower', extent=[xmin, xmax, 0, spectrum.shape[0]])

    # making the x axis showing time values as HH:MM:SS
    fig.get_figure().autofmt_xdate()
    ax = fig.get_axes()
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(SecondLocator(interval=300))
    ax.xaxis.set_minor_locator(SecondLocator(interval=60))


def main():
    name = '4H_Community_Guest_Mix_The_2nd_Anniversary_of_Room51_Show_by_Breeze_Quadrat_PureFM'
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

    filtered = tools.detect_track_borders(data,
                                          timestamps[-1],
                                          len(info),
                                          self_sim=self_sim,
                                          factor=factor,
                                          has_intro=has_intro,
                                          has_outro=has_outro)
    for i in range(len(info)):
        print '%d.\t%f\t%f: %f' % (i, info[i]['INDEX'], filtered[i], filtered[i] - info[i]['INDEX'])
    print filtered
    print [x['INDEX'] for x in info]
    print experiment.get_diff(info, filtered)

    data = data.transpose((1, 0))
    draw_spectrum(data, timestamps, 1)
    # borders = [x * sec_per_row for x in filtered]
    vlines([datetime.datetime.fromtimestamp(x) for x in filtered], 0, 48, color='k', linewidths=[2]*len(filtered))
    vlines([datetime.datetime.fromtimestamp(x['INDEX']) for x in info], 0, 48, color='c', linewidths=[2]*len(info))

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


if __name__ == '__main__':
    main()
