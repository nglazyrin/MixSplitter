__author__ = 'Nikolay'

import cPickle
import csv
import numpy
import os
import subprocess
import tools


plugin_suffix = '_vamp_qm-vamp-plugins_qm-constantq_constantq'
vamp_path = 'E:/Study/sci-work/Software/sonic-annotator-1.0-win32/sonic-annotator.exe'
n3_path = 'E:/Study/sci-work/Software/sonic-annotator-1.0-win32/constantq.n3'


def read_csv(filename):
    timestamps = []
    data = []
    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            timestamps.append(float(row[0]))
            data.append([float(x) for x in row[1:]])
    data = numpy.array(data)
    timestamps = numpy.array(timestamps)
    return timestamps, data


def read_cue(filename):
    info = []
    current = {}
    with open(filename, 'rb') as input_file:
        for line in input_file:
            line = line.strip()
            if line.startswith('TRACK'):
                current = {'PERFORMER': '', 'TITLE': '', 'INDEX': 0}
                info.append(current)
            if line.startswith('INDEX'):
                (m, s, ds) = line.split()[2].split(':')
                current['INDEX'] = int(m) * 60 + int(s) + 0.01 * int(ds)
            if line.startswith('PERFORMER'):
                current['PERFORMER'] = line.split('"')[1]
            if line.startswith('TITLE'):
                current['TITLE'] = line.split('"')[1]
    # if 'intro' in info[0]['TITLE'].lower():
    #     info = info[1:]
    #     info[0]['INDEX'] = 0
    return info


def extract_features(mp3):
    mp3_path = os.path.abspath(mp3).replace('\\', '/')
    retcode = subprocess.call(vamp_path + ' -t ' + n3_path + ' "' + mp3_path + '" -w csv')
    if retcode != 0:
        print 'Feature extraction process exited with return code %d, skipping the %s' % (retcode, mp3)


def get_diff(cue_info, borders):
    max_diff = 0
    avg_diff = 0
    for i in range(len(cue_info)):
        a = cue_info[i]['INDEX']
        b = borders[i]
        avg_diff += abs(a - b)
        max_diff = max(abs(a - b), max_diff)

        def sec_to_colon_separated(seconds):
            hrs = int(seconds / 3600)
            mins = int((seconds % 3600) / 60)
            secs = seconds % 60
            return '%02d:%02d:%02d' % (hrs, mins, secs)
        #print '%d. %s - %s' % (i + 1, sec_to_colon_separated(a), sec_to_colon_separated(b))
    return avg_diff / len(cue_info), max_diff


def print_results(results, log):
    print '%d mixes have been processed' % (len(results))
    log.write('%d mixes have been processed\r\n' % (len(results)))
    avg = 0
    for result in results:
        avg += result['avg_diff']
    print 'Average difference from CUE file: %f' % (avg / len(results))
    log.write('Average difference from CUE file: %f\r\n' % (avg / len(results)))
    # for result in results:
    #     print '\t%s: avg_diff: %f, max_diff: %f' % (result['name'], result['avg_diff'], result['max_diff'])


def main():
    data_dir = os.path.join('.', 'data')
    mp3_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.mp3')]
    result = []
    with open(os.path.join('logs', 'test.log'), 'wb') as log:
        for mp3 in mp3_files:
            name = mp3[:-4]
            cue = os.path.join(data_dir, name + '.cue')
            info = None
            if os.path.isfile(cue):
                info = read_cue(cue)
            else:
                print 'No cue file for %s, skipping' % mp3
            if info:
                has_intro = 'intro' in info[0]['TITLE'].strip().lower()
                has_outro = 'outro' in info[-1]['TITLE'].strip().lower()
                has_intro = False
                has_outro = False
                csv_file = os.path.join(data_dir, name + plugin_suffix + '.csv')
                self_sim_file = os.path.join(data_dir, name + plugin_suffix + '.sim')
                if not os.path.isfile(csv_file):
                    extract_features(os.path.join(data_dir, mp3))
                if os.path.isfile(csv_file):
                    (timestamps, data) = read_csv(csv_file)
                    if os.path.isfile(self_sim_file):
                        with open(self_sim_file, 'rb') as f:
                            (self_sim, factor) = cPickle.load(f)
                        borders = tools.detect_track_borders(data,
                                                             timestamps[-1],
                                                             len(info),
                                                             self_sim=self_sim,
                                                             factor=factor,
                                                             has_intro=has_intro,
                                                             has_outro=has_outro)
                    else:
                        borders = tools.detect_track_borders(data,
                                                             timestamps[-1],
                                                             len(info),
                                                             sim_file=self_sim_file,
                                                             has_intro=has_intro,
                                                             has_outro=has_outro)
                    # borders = tools.detect_track_borders(data, timestamps[-1], len(info))
                    avg_diff, max_diff = get_diff(info, borders)
                    result.append({'name': mp3, 'info': info, 'borders': borders, 'avg_diff': avg_diff, 'max_diff': max_diff})
                    print '%s\t%f\t%f' % (mp3, avg_diff, max_diff)
                    log.write('%s\t%f\t%f\t%s\r\n' % (mp3, avg_diff, max_diff, borders))
        print_results(sorted(result, key=lambda x: x['name']), log)


if __name__ == '__main__':
    main()