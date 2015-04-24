__author__ = 'Nikolay'

import cPickle
import numpy
import numpy.linalg
import scipy.signal
import operator
from scipy.spatial.distance import cosine
from scipy.signal import argrelextrema

from matplotlib.pylab import *
from scipy.stats import norm
from sklearn.preprocessing import normalize
import matplotlib.dates as mdates


class NoveltyCalculator:
    def __init__(self):
        self.checkerboard_size = 8
        self.checkerboard = NoveltyCalculator.get_checkerboard(self.checkerboard_size)

    @staticmethod
    def pdf_multivariate_gauss(x, mu, cov_1, part1):
        '''
        Caculate the multivariate normal density (pdf)

        Keyword arguments:
            x = numpy array of a "d x 1" sample vector
            mu = numpy array of a "d x 1" mean vector
            cov = inverted "numpy array of a d x d" covariance matrix
        '''
        # assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
        # assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
        # assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
        # assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
        # assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
        d = x - mu
        part2 = (-1/2) * (d.T.dot(cov_1)).dot(d)
        return float(part1 * np.exp(part2))

    @staticmethod
    def get_checkerboard(cb_size):
        o = numpy.ones([cb_size, cb_size])
        xs = np.linspace(-1, 1, 2 * cb_size)
        ys = np.linspace(-1, 1, 2 * cb_size)
        result = numpy.array(numpy.bmat([[-o, o], [o, -o]]))
        mu = numpy.array([0, 0])
        covariance_matrix = numpy.array([[1, 0], [0, 1]])
        inverted_covariance_matrix = np.linalg.inv(covariance_matrix)
        part1 = 1 / ( ((2*np.pi)**(len(mu)/2)) * (np.linalg.det(covariance_matrix)**(1/2)) )
        ix = 0
        for x in xs:
            iy = 0
            for y in ys:
                value = NoveltyCalculator.pdf_multivariate_gauss(numpy.array([x, y]), mu,
                                                                 inverted_covariance_matrix, part1)
                result[ix][iy] *= value
                iy += 1
            ix += 1
        return result

    def calc_novelty(self, self_similarity_matrix):
        novelty = []
        self_sim_size = self_similarity_matrix.shape[0]
        for i in range(self_sim_size):
            if i < self.checkerboard_size:
                r = numpy.ix_(range(self.checkerboard_size - i, 2 * self.checkerboard_size),
                              range(self.checkerboard_size - i, 2 * self.checkerboard_size))
                cb = self.checkerboard[r]
                ssr = numpy.ix_(range(0, cb.shape[0]), range(0, cb.shape[0]))
            elif i > self_sim_size - self.checkerboard_size:
                d = self_sim_size - i
                r = numpy.ix_(range(0, self.checkerboard_size + d), range(0, self.checkerboard_size + d))
                cb = self.checkerboard[r]
                ssr = numpy.ix_(range(self_sim_size - cb.shape[0], self_sim_size),
                                range(self_sim_size - cb.shape[0], self_sim_size))
            else:
                cb = self.checkerboard
                ssr = numpy.ix_(range(i - self.checkerboard_size, i + self.checkerboard_size),
                                range(i - self.checkerboard_size, i + self.checkerboard_size))
            fragment = self_similarity_matrix[ssr]
            sq = scipy.signal.convolve2d(cb, fragment, mode='valid')
            novelty.append(sq[0][0])
        return novelty


class Segment():
    def __init__(self, start, end, feature_vector):
        self.start = start
        self.end = end
        self.length = end - start
        self.feature_vector = feature_vector
        self.normalized_feature_vector = numpy.array(feature_vector, copy=True)
        normalize_to_01(self.normalized_feature_vector)


def distance(segment1, segment2):
    #return numpy.linalg.norm(segment1.normalized_feature_vector - segment2.normalized_feature_vector)
    return cosine(segment1.feature_vector, segment2.feature_vector)


def key_function(segment1, segment2, max_length_with_no_penalty):
    sum_length = segment2.end - segment1.start
    segments_distance = distance(segment1, segment2)
    if sum_length < max_length_with_no_penalty:
        return sum_length * segments_distance  # was: first_length * sum_length * segments_distance
    return sum_length * 16


def join_segments(segments, start, end, distances, sec_per_row, avg_track_length, data, factor):
    '''
    Both ends are included
    :param segments: list of consecutive segments
    :param start: position of the first segment
    :param end: position of the last segment (inclusive)
    :param distances: list of tuples (segment1, segment2, distance)
    :return: nothing, all updates are performed in-place
    '''
    deleted_segments = set(segments[start:end+1])
    distances[:] = [x for x in distances if not (x[0] in deleted_segments or x[1] in deleted_segments)]

    # new_feature_vector = segments[start].feature_vector * segments[start].length
    # for k in range(start + 1, end + 1):
    #     new_feature_vector += (segments[k].feature_vector * segments[k].length)
    # new_feature_vector /= (segments[end].end - segments[start].start)
    new_feature_vector = get_feature_vector(data, segments[start].start * factor, segments[end].end * factor)
    new_segment = Segment(segments[start].start, segments[end].end, new_feature_vector)

    del(segments[start:end+1])
    segments.insert(start, new_segment)

    window = 600.0 / sec_per_row
    max_sum_length = avg_track_length * 1.25 / sec_per_row
    new_segment_start = new_segment.start
    for segment in segments:
        if segment != new_segment and abs(segment.start - new_segment_start) < window:
            if segment.start > new_segment_start:
                distances.append((new_segment, segment, key_function(new_segment, segment, max_sum_length)))
            else:
                distances.append((segment, new_segment, key_function(segment, new_segment, max_sum_length)))


def get_distances(segments, sec_per_row, avg_track_length):
    distances = []
    max_sum_length = avg_track_length * 1.25 / sec_per_row
    for i in range(len(segments)):
        segment1 = segments[i]
        for j in range(i+1, len(segments)):
            segment2 = segments[j]
            distances.append((segment1, segment2, key_function(segment1, segment2, max_sum_length)))
    return distances


def get_initial_borders(novelty):
    borders = argrelextrema(numpy.array(novelty), numpy.greater)
    borders = numpy.insert(borders, 0, 0)
    borders = numpy.append(borders, len(novelty))
    return borders


def get_segments(borders, feature_vectors):
    return [Segment(borders[i], borders[i+1], feature_vectors[i]) for i in range(len(borders) - 1)]


def get_borders(segments):
    borders = [segments[0].start]
    borders.extend([segment.end for segment in segments])
    return borders


def calc_self_similarity(data, factor):
    # reduce data size to calculate self-similarity
    data10 = [numpy.average(data[i * factor: (i + 1) * factor], axis=0) for i in xrange(data.shape[0] / factor)]
    data10 = numpy.array(data10)
    selfsim = numpy.zeros([data10.shape[0], data10.shape[0]])
    for i in xrange(data10.shape[0]):
        left = max(0, i - 300)
        right = min(data10.shape[0], i + 300)
        for j in range(left, right):
            selfsim[i][j] = selfsim[j][i] = cosine(data10[i], data10[j])
    return selfsim


def normalize_to_01(x):
    m = numpy.max(numpy.abs(x))
    if m != 0:
        x /= m


def get_feature_vector(data, start, end):
    subset = data[start: end]
    vector1 = numpy.mean(subset, axis=0)
    normalize_to_01(vector1)

    energy = numpy.sum(subset, axis=1)
    local_max = argrelextrema(energy, numpy.greater)[0]
    local_min = argrelextrema(energy, numpy.less)[0]
    avg_max = numpy.average(energy[local_max])
    avg_min = numpy.average(energy[local_min])
    i = 0
    j = 0
    local_attacks = {'slow': [], 'fast': []}
    local_decays = {'slow': [], 'fast': []}
    while i < len(local_max) and j < len(local_min):
        value = (energy[local_max[i]] - energy[local_min[j]]) / abs(local_max[i] - local_min[j])
        peak_type = 'fast' if energy[local_max[i]] > avg_max else 'slow'
        if local_max[i] > local_min[j]:
            local_attacks[peak_type].append(value)
            j += 1
        else:
            local_decays[peak_type].append(value)
            i += 1
    vector2 = numpy.array([#numpy.mean(local_attacks['slow']), numpy.std(local_attacks['slow']),
                           numpy.mean(local_attacks['fast']), numpy.std(local_attacks['fast']),
                           #numpy.mean(local_decays['slow']), numpy.std(local_decays['slow']),
                           numpy.mean(local_decays['fast']), numpy.std(local_decays['fast']),
                           avg_max / avg_min])
    '''
    attacks = []
    decays = []
    for row in range(subset.shape[1]):
        # energy = numpy.sum(subset, axis=1)
        energy = subset[:, row]
        local_max = argrelextrema(energy, numpy.greater)[0]
        local_min = argrelextrema(energy, numpy.less)[0]
        local_attacks = 0
        local_decays = 0
        while i < len(local_max) and j < len(local_min):
            value = (energy[local_max[i]] - energy[local_min[j]]) / abs(local_max[i] - local_min[j])
            if local_max[i] > local_min[j]:
                local_attacks += value
                j += 1
            else:
                local_decays += value
                i += 1
        attacks.append(local_attacks)
        decays.append(local_decays)
    '''
    # vector2 = numpy.array(attacks)
    normalize_to_01(vector2)
    # vector3 = numpy.array(decays)
    # normalize_to_01(vector3)
    return numpy.concatenate((vector1, vector2), axis=0)
    # return numpy.concatenate((vector1, vector2, vector3), axis=0)
    # return vector2


def get_feature_vectors(data, factor, borders):
    vectors = []
    for i in range(len(borders) - 1):
        start = borders[i] * factor
        end = borders[i+1] * factor
        vectors.append(get_feature_vector(data, start, end))
    # energy = numpy.sum(data[borders[4] * factor: borders[5] * factor], axis=1)
    # plt.plot(energy)
    # plt.show()
    return vectors


def detect_track_borders(data, length_in_sec, tracks, novelty_calculator, self_sim=None, factor=None, sim_file=None,
                         has_intro=False, has_outro=False, preprocess=True):
    if preprocess:
        data = numpy.log(1e6 * data + 1)
        # data = scipy.signal.medfilt2d(data, kernel_size=(31, 1))
    if self_sim is None or factor is None:
        factor = 80
        self_sim = calc_self_similarity(data, factor)
        if sim_file:
            numpy.savez(sim_file, self_sim=self_sim, factor=factor)
            # with open(sim_file, 'wb') as f:
                # cPickle.dump((self_sim, factor), f)

    # remove segments which are too short
    sec_per_row = length_in_sec / self_sim.shape[0]
    avg_track_length = length_in_sec / tracks
    novelty = novelty_calculator.calc_novelty(self_sim)
    borders = get_initial_borders(novelty)

    if has_intro:
        first_60_seconds = borders[borders < 60.0 / sec_per_row]
        max_position = numpy.argmax(first_60_seconds)
        borders = borders[max_position:]
        tracks -= 1
    if has_outro:
        last_60_seconds = borders[borders > self_sim.shape[0] - 60.0 / sec_per_row]
        max_position = len(borders) - len(last_60_seconds) + numpy.argmax(last_60_seconds)
        borders = borders[:max_position + 1]
        tracks -= 1

    if preprocess:
        feature_vectors = get_feature_vectors(data, factor, borders)
    else:
        feature_vectors = data

    segments = get_segments(borders, feature_vectors)
    distances = get_distances(segments, sec_per_row, avg_track_length)
    segment_indices = dict((x, i) for i, x in enumerate(segments))

    image_data = numpy.zeros([len(segments), len(segments)])
    for d in distances:
        x = segment_indices[d[0]]
        y = segment_indices[d[1]]
        value = math.log10(1 + d[2])
        image_data[x, y] = image_data[y, x] = value
    #plt.imshow(image_data, interpolation='nearest')
    #plt.show()

    while len(segments) > tracks:
        distances = sorted(distances, key=operator.itemgetter(2))
        for d in distances:
            index1 = segment_indices[d[1]]
            index0 = segment_indices[d[0]]
            if index0 + len(segments) - index1 >= tracks:
                break
        join_segments(segments, index0, index1, distances, sec_per_row, avg_track_length, data, factor)
        segment_indices = dict((x, i) for i, x in enumerate(segments))

    borders = get_borders(segments)
    if has_intro:
        borders.insert(0, 0)
    if has_outro:
        borders.append(self_sim.shape[0])

    return [x * sec_per_row for x in borders]
