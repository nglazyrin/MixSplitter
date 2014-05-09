__author__ = 'Nikolay'

import cPickle
import numpy
import numpy.linalg
import scipy.signal
from scipy.spatial.distance import cosine

from matplotlib.pylab import *
from scipy.stats import norm
from sklearn.preprocessing import normalize
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def remove_short_segments(data, borders, factor, sec_per_row):
    filtered = set(borders)
    i = 1
    while i < len(borders) - 2:
        if (borders[i + 1] - borders[i]) * sec_per_row < 30:
            avg_1 = numpy.average(data[borders[i-1]*factor: borders[i]*factor], axis=0)
            avg = numpy.average(data[borders[i]*factor: borders[i+1]*factor], axis=0)
            avg1 = numpy.average(data[borders[i+1]*factor: borders[i+2]*factor], axis=0)
            lim = 0.05
            d_from_next = numpy.linalg.norm(avg1 - avg)
            d_next_from_prev = numpy.linalg.norm(avg1 - avg_1)
            if d_next_from_prev < lim:
                if borders[i] in filtered:
                    filtered.remove(borders[i])
                filtered.remove(borders[i+1])
                i += 2
            elif d_from_next < lim:
                filtered.remove(borders[i+1])
                i += 1
            # elif d_next_from_prev < d_from_prev:
            #     if borders[i] in filtered:
            #         filtered.remove(borders[i])
            #     filtered.remove(borders[i+1])
            # elif d_from_prev < d_next_from_prev:
            #     filtered.remove(borders[i+1])
        i += 1
    return sorted(list(filtered))


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


def get_checkerboard(size):
    o = numpy.ones([size, size])
    xs = np.linspace(-1, 1, 2 * size)
    ys = np.linspace(-1, 1, 2 * size)
    result = numpy.array(numpy.bmat([[-o, o], [o, -o]]))
    mu = numpy.array([0, 0])
    cov = numpy.array([[1, 0], [0, 1]])
    cov_1 = np.linalg.inv(cov)
    part1 = 1 / ( ((2*np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    ix = 0
    for x in xs:
        iy = 0
        for y in ys:
            value = pdf_multivariate_gauss(numpy.array([x, y]), mu, cov_1, part1)
            result[ix][iy] *= value
            iy += 1
        ix += 1
    return result

def calc_self_similarity(data, factor):
    # reduce data size to calculate self-similarity
    data10 = []
    for i in xrange(data.shape[0] / factor):
        temp = data[i * factor: (i + 1) * factor]
        data10.append(numpy.average(temp, axis=0))
    data10 = numpy.array(data10)
    # data10 = data
    # factor = 1
    selfsim = numpy.zeros([data10.shape[0], data10.shape[0]])
    for i in xrange(data10.shape[0]):
        left = max(0, i - 300)
        right = min(data10.shape[0], i + 300)
        for j in range(left, right):
            selfsim[i][j] = selfsim[j][i] = cosine(data10[i], data10[j])
    return selfsim


def remove_most_similar(data, borders, factor, sec_per_row, novelty, avg_track_length):
    filtered = set(borders)
    i = 1
    similar = []
    bl = len(borders)
    novelties = numpy.array([novelty[borders[i]] for i in range(bl-1)])
    avg_novelty = numpy.average(novelties)

    # separate handling for the very first and the very last segments
    # if (borders[1] - borders[0]) * sec_per_row < 10:
    #     similar.append((borders[0], borders[1], borders[1] - borders[0], 0.001))
    # if (borders[bl-1] - borders[bl-2]) * sec_per_row < 10:
    #     similar.append((borders[bl-2], borders[bl-1], borders[bl-1] - borders[bl-2], 0.001))

    def normalize_to_01(x):
        m = numpy.max(numpy.abs(x))
        if m == 0:
            m = 1
        x /= m

    for i in range(bl - 1):
        l1 = borders[i]
        r1 = borders[i+1]
        source = numpy.average(data[l1 * factor: r1 * factor], axis=0)
        normalize_to_01(source)
        nov = 0
        for j in range(i + 1, bl - 1):
            nov += novelty[borders[j]]
            if abs(borders[j] - l1) * sec_per_row < 600:
                l2 = borders[j]
                r2 = borders[j+1]
                target = numpy.average(data[l2 * factor: r2 * factor], axis=0)
                normalize_to_01(target)
                similar.append((l1, r1, l2, r2, numpy.linalg.norm(source - target), nov*1.0/(j-i)))

    def key_function(t):
        # t == (left1, right1, left2, right2, distance, novelty)
        sum_length = t[3]-t[0]
        first_length = t[1]-t[0]
        step = t[2] - t[0]
        # if l1 == 0 and 5 < d * sec_per_row < 60:
        #     return 100000
        distance = t[4]
        if sum_length * sec_per_row > avg_track_length * 1.25:
            return 100000 * sum_length
        return sum_length * distance # TODO: was first_length * sum_length * distance

    similar = sorted(similar, key=key_function)
    min = similar[0][0]
    max = similar[0][2]
    filtered = [x for x in borders if x <= min or x > max]
    # print [x * sec_per_row for x in filtered]
    return sorted(filtered)
    #return filtered


def init_borders2(self_sim):
    size = 8
    checkerboard = get_checkerboard(size)
    i = 0
    squareness = []
    borders = [0]
    self_sim_size = self_sim.shape[0]
    for i in range(self_sim_size):
        if i < size:
            r = numpy.ix_(range(size - i, 2 * size), range(size - i, 2 * size))
            cb = checkerboard[r]
            ssr = numpy.ix_(range(0, cb.shape[0]), range(0, cb.shape[0]))
        elif i > self_sim.shape[0] - size:
            d = self_sim.shape[0] - i
            r = numpy.ix_(range(0, size + d), range(0, size + d))
            cb = checkerboard[r]
            ssr = numpy.ix_(range(self_sim_size - cb.shape[0], self_sim_size), range(self_sim_size - cb.shape[0], self_sim_size))
        else:
            cb = checkerboard
            ssr = numpy.ix_(range(i - size, i + size), range(i - size, i + size))
        fragment = self_sim[ssr]
        sq = scipy.signal.convolve2d(cb, fragment, mode='valid')
        squareness.append(sq[0][0])
    for i in range(self_sim_size):
        if 0 < i < self_sim_size - 1 and squareness[i] > squareness[i-1] and squareness[i] > squareness[i+1]:
            borders.append(i)
    borders.append(self_sim_size)
    return borders, squareness


def detect_track_borders(data, length_in_sec, tracks, self_sim=None, factor=None, sim_file=None, has_intro=False, has_outro=False):
    data = log(data)
    data = scipy.signal.medfilt2d(data, kernel_size=(31, 1))
    if self_sim is None or factor is None:
        factor = 10
        self_sim = calc_self_similarity(data, factor)
        if sim_file:
            with open(sim_file, 'wb') as f:
                cPickle.dump((self_sim, factor), f)

    # remove segments which are too short
    sec_per_row = length_in_sec / self_sim.shape[0]
    avg_track_length = length_in_sec / tracks
    filtered, novelty = init_borders2(self_sim)

    intro_borders = []
    outro_borders = []
    if has_intro:
        max_novelty = 0
        max_position = 1
        i = max_position
        while filtered[i] * sec_per_row <= 60:
            if novelty[i] > max_novelty:
                max_novelty = novelty[i]
                max_position = i
            i += 1
        intro_borders = [0]
        filtered = filtered[max_position:]
        tracks -= 1
    if has_outro:
        max_novelty = 0
        max_position = len(filtered) - 2
        i = max_position
        while (self_sim.shape[0] - filtered[i]) * sec_per_row <= 60:
            if novelty[i] > max_novelty:
                max_novelty = novelty[i]
                max_position = i
            i -= 1
        outro_borders = [self_sim.shape[0]]
        filtered = filtered[:max_position + 1]
        tracks -= 1

    while len(filtered) > tracks + 1:
        filtered = remove_most_similar(data, filtered, factor, sec_per_row, novelty, avg_track_length)
    intro_borders.extend(filtered)
    if has_outro:
        intro_borders.append(self_sim.shape[0])
    filtered = intro_borders

    filtered = [x * sec_per_row for x in filtered]
    return filtered

