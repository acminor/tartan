#!/usr/bin/env python3

import utils
import click
import toml
import itertools
import numpy as np
import numpy.fft as fft
import scipy.signal as sig
import scipy.optimize as opti
import matplotlib.pyplot as plt
import msgpack
import math
from pathlib import Path
from joblib import Parallel, delayed
from time import time
from datetime import datetime
from multiprocessing import cpu_count, Value, Manager
from collections import namedtuple
import sqlite3

class NewDataWriter:
    def connect(self):
        class Connection:
            def __init__(self, lock, db):
                self._lock = lock
                self._db = db
            def __enter__(self):
                return self._db
            def __exit__(self, type, value, traceback):
                self._db.commit()
                self._db.close()
                self._lock.release()

        while True:
            for i, lock in enumerate(self._db_locks):
                if lock.acquire(blocking=False):
                    self._dbs[i] = sqlite3.connect(self._get_db_name(i))
                    return Connection(self._db_locks[i], self._dbs[i])
    def close(self):
        conn = sqlite3.connect(self._get_db_name(0))
        c = conn.cursor();

        for i in range(1, self._num_dbs):
            c.executescript(
                "attach \"{}\" as toMerge;".format(self._get_db_name(i)) +
                "insert into StarEntry select * from toMerge.StarEntry;"
                "detach toMerge;",
            )
            conn.commit()

        c.execute(
            "CREATE TABLE IF NOT EXISTS GenSchema("
            "   run_time TIMESTAMP PRIMARY KEY,"
            "   schema TEXT NOT NULL"
            ");"
        )

        c.execute(
            "insert into GenSchema(run_time, schema) VALUES (?, ?);",
            (datetime.utcnow(), toml.dumps(self._gen_toml_desc))
        )

        conn.commit()
        conn.close()

        Path(self._get_db_name(0)).rename(self._db_file_name)
        for i in range(1, self._num_dbs):
            Path(self._get_db_name(i)).unlink()
    def _put_schema(self, db):
        conn = sqlite3.connect(db)
        c = conn.cursor()
        c.execute("PRAGMA cache_size = -20240;")
        c.execute("PRAGMA journal_mode = wal;")
        c.execute(
            "CREATE TABLE IF NOT EXISTS StarEntry("
            "   id INTEGER PRIMARY KEY,"
            "   desc TEXT NOT NULL,"
            "   data BLOB NOT NULL"
            ");"
        )
        conn.commit()
        conn.close()
    def _get_db_name(self, i):
        return self._db_file_name + '{}'.format(i)
    def __init__(self, db_file_name, lock_factory, num_dbs, gen_toml_desc):
        self._db_file_name = db_file_name
        self._db_locks = [lock_factory() for i in range(0, num_dbs)]
        self._dbs = [None for i in range(0, num_dbs)]
        self._num_dbs = num_dbs
        self._gen_toml_desc = gen_toml_desc

        for i in range(0, num_dbs):
            self._put_schema(self._get_db_name(i))

# NOTE use mean power since white noise
def noise_power(noise, sample_rate):
    nperseg = 1024
    if len(noise) < nperseg:
        nperseg = len(noise)

    return 10*np.log10(np.sum(sig.welch(noise, sample_rate, nperseg=nperseg)[1]))

# NOTE use max power since one time occurring signal
#      and the signal is constant for each parameter set
def signal_power(signal, sample_rate):
    nperseg = 1024
    if len(signal) < nperseg:
        nperseg = len(signal)

    avg = np.max(sig.welch(signal, sample_rate, nperseg=nperseg)[1])
    power = 10*np.log10(avg)
    return power

@click.command()
@click.option('-o', '--output',
              help='directory for old_data_fmt, sqlite3 db file for new_data_fmt',
              type=click.Path(),
              required=True)
@click.option('-i', '--desc-file', type=click.Path(), required=True)
@click.option('-plot', type=bool, default=False)
@click.option('-new-data-fmt', type=bool, default=False)
def main(output, desc_file, plot, new_data_fmt):
    with open(desc_file) as a:
        desc_data = toml.load(a)

    if not plot:
        if not new_data_fmt and len(list(Path(output).glob('*'))) != 0:
            if click.confirm('Output dir is not empty. Empty?'):
                for f in Path(output).glob('*'):
                    f.unlink()
        elif new_data_fmt and Path(output).exists():
            if click.confirm('Output db is not empty. Empty?'):
                Path(output).unlink()

    config_name = Path(desc_file).stem
    stars = _gen(desc_data, config_name, output, plot, new_data_fmt)

def write_star(star, config_name, output_dir, plot, star_count, new_data_writer):
    path_name = (
        "config={},".format(config_name) +
        "len={},".format(len(star['samples'])) +
        "spow={:.2f},".format(star['spow']) +
        "npow={:.2f},".format(star['npow']) +
        "snr={:.2f},".format(star['snr']) +
        "tl={},".format(star['tl']) +
        "tr={}".format(star['tr'])
    )

    if plot:
        plt.plot(star['samples'])
        plt.gca().invert_yaxis()
        plt.title(path_name)
        plt.show()
    else:
        star['id'] = path_name
        star['star_type'] = 'unknown'
        star['sample_rate'] = 15
        star['arima_model_file'] = ''

        def get_msgpack():
            return msgpack.packb(list(star['samples']))

        # hopefully to prevent two files from being the same
        with star_count.lock:
            i = star_count.value.value
            star_count.value.value += 1

        if new_data_writer is not None:
            with new_data_writer.connect() as conn:
                c = conn.cursor()

                data = get_msgpack()

                star['samples'] = ''
                desc = toml.dumps(star)

                c.execute(
                    "INSERT INTO StarEntry(id, desc, data) VALUES (?, ?, ?)",
                    (i, desc, data)
                )
        else:
            with open(str(Path(output_dir)/Path('star' + str(i) + '.mpk')), 'wb+') as file:
                file.write(get_msgpack())

            star['samples'] = 'star' + str(i) + '.mpk'

            with open(str(Path(output_dir)/Path('star' + str(i) + '.toml')), 'w+') as file:
                toml.dump(star, file)

def sine(period, amplitude, phase):
    return lambda t: amplitude*math.sin((2*math.pi*(t/period) + np.radians(phase)) % (2*math.pi))

def cosine(period, amplitude, phase):
    return lambda t: amplitude*math.cos((2*math.pi*(t/period) + np.radians(phase)) % (2*math.pi))

def get_flare_star_v2(amplitude, tot_time, _sample_time):
    '''
    Flare light curve modeled using observations of GJ1234.

    Kepler Flares. II. The Temporal Morphology of White-light Flares on GJ 1234
    Davenport, et. al.
    The Astrophysical Journal, 797:122 (Dec. 20, 2014)

    t_half is special index that is a relative index.
    - sampling into it (i.e. further subdividing) t_half
      gets different lengths of events
    amplitude
    - scale for the final amplitude at peak (peak is by default 1.0)
    '''

    # [ ] TODO double check formulations

    def calc(t_half):
        def rise(t_half):
            return 1 + 1.941*t_half - 0.175*np.power(t_half, 2.0) \
                - 2.246*np.power(t_half, 3.0) - 1.125*np.power(t_half, 4.0)
        def fall(t_half):
            return 0.6890*np.exp(-1.600*t_half) + 0.3030*np.exp(-0.2783*t_half)

        if -1.0 <= t_half and t_half <= 0.0:
            return rise(t_half)
        elif 0.0 < t_half and t_half <= 6.0:
            return fall(t_half)
        else: # end at 7.0 w/ linear decay to 0.0
            start = fall(6.0)
            end = 0.0
            m = (start - end)/(6.0 - 7.0)
            # b = y - mx
            b = fall(6.0) - m*6.0
            # y = mx + b
            return m*t_half + b

    # [ ] TODO double check time and generation logic
    def gen():
        data = []
        step = 8.0 / (tot_time / float(_sample_time))
        # plus step to ensure that we completely cover 7.0
        # and thus end at 0.0
        for t_half in np.arange(-1.0, 7.0 + step, step):
            data.append(amplitude*calc(t_half))

        # some data generation reaches 0.0 faster than others
        output = []
        for d in data:
            if d < 0.0:
                output.append(0.0)
            else:
                output.append(d)

        return data

    return gen

# NOTE: FOR NOW NOT USABLE
def get_flare_star(A, B, C, D, E, F):
    '''
    Flare light curve modeled using soft X-Rays from

    Model of flare lightcurve profile observed in soft X-rays
    Magdalena Gryciuk, et.al.
    Proceedings IAU Symposium No. 320, 2015
    A.G. Kosovichev, S.L. Hawley & P. Heinzel, eds.

    A, B, C, D are flare parameters
    E, F are linear background parameters
    -> seems to model something associated
       - [ ] look up SXR emission
    '''

    # TODO [ ] is the final form of the flare f(t) correct???
    def calc(t):
        a = (1.0/2.0)*math.sqrt(math.pi)*A*C
        b = math.exp(D*(B - t) + ((C**2)*(D**2))/(4.0))
        Z = (2*B + (C**2)*D)/(2*C)
        # TODO [ ] check the error function to make sure the
        #          same as used in the paper
        c = (math.erf(Z) - math.erf(Z - (t/C)))
        flare = a*b*c
        dc = E*t + F

        return flare + dc

    return lambda t: calc(t)

def parse_range(ran):
    LOW = 0
    HIGH = 1
    STEP = 2

    if type(ran) != list:
        return [ran]

    if len(ran) == 1:
        return [ran[0]]
    elif len(ran) == 2:
        return list(np.arange(ran[LOW], ran[HIGH], dtype=float))
    elif len(ran) == 3:
        return list(np.arange(ran[LOW], ran[HIGH], ran[STEP], dtype=float))
    else:
        print("ERROR: BAD RANGE")
        exit(-1)

def incremental_sum(a):
    b = np.zeros(len(a))
    for i in range(0, len(a)):
        if i == 0:
            b[i] = a[i]
        else:
            b[i] = a[i] + b[i-1]
    return b

# Their are arguments to be made about a constant (small-mag) phase
# wobble with occasional bumps and for not correcting back to 0 (full-correction)
# on the next hit. Yet, we also see this as a good model to test both the effect
# of short-term amplitude mispredictions and to test long-term phase mispredictions
#
# clarification: full-correction back to localized 0 (starting phase)
#
# their is also something to be said about gaussian noise layered on top of the raw
# signal may affect the prediction differently than just as if the signal is shifted
# that gaussian noise level up or down. (we do not consider this)
def sine_with_phase_error(period, amplitude, starting_phase,
                          mean_phase, var_phase, dur_samps_phase, sample_rate):
    MIN_ANOMALY_VALUES = 10000
    dur_between_phase_anomaly = np.floor(np.random.power(1, size=MIN_ANOMALY_VALUES)*dur_samps_phase)
    phase_anomaly_points = incremental_sum(dur_between_phase_anomaly)

    # FIXME same as 'spe'
    if dur_samps_phase <= 1.0:
        flip_or_shift = np.zeros(MIN_ANOMALY_VALUES)
        print(
            'WARNING: switching phase every turn is essentially amplitude distortion (duration <= 1.0)'
        )
        for i in range(0, MIN_ANOMALY_VALUES):
            flip_or_shift[i] = True
    else:
        flip_or_shift = np.zeros(int(phase_anomaly_points[-1]) + 1)
        for point in phase_anomaly_points:
            flip_or_shift[int(point)] = True

    data = {}
    data['current_phase_shift'] = 0.0
    def inner_function(t):
        # due to inexact floating point, this should be even
        # but we round to the nearest because it might not be
        # so ceiling and floor would not be appropriate
        index = int(np.around(t/sample_rate))

        if flip_or_shift[index]:
            if data['current_phase_shift'] != 0.0:
                data['current_phase_shift'] = 0.0
            else:
                data['current_phase_shift'] = np.random.normal(loc=mean_phase, scale=var_phase)

        part_a = 2*math.pi*(t/period)
        part_b = np.radians(starting_phase + data['current_phase_shift'])
        return amplitude*math.sin((part_a + part_b) % (2*math.pi))

    return inner_function

def parse_noise_descriptors(descriptors):
    global_funcs = list()

    def parse_descriptor(desc):
        if desc['type'] == 'sine':
            periods = parse_range(desc['period'])
            phases = parse_range(desc['phase'])
            amplitudes = parse_range(desc['amplitude'])
            args = itertools.product(periods, amplitudes, phases)
            funcs = [sine(period, amplitude, phase) for (period, amplitude, phase) in args]
            return funcs
        elif desc['type'] == 'cosine':
            periods = parse_range(desc['period'])
            amplitudes = parse_range(desc['amplitude'])
            phases = parse_range(desc['phase'])
            args = itertools.product(periods, amplitudes, phases)
            funcs = [cosine(period, amplitude, phase) for (period, amplitude, phase) in args]
            return funcs
        elif desc['type'] == 'gaussian':
            def get_gaussian(variance, mean):
                return lambda t: np.random.normal(scale=variance, loc=mean, size=1)[0]
            means = parse_range(desc['mean'])
            variances = parse_range(desc['variance'])
            args = itertools.product(means, variances)
            funcs = [get_gaussian(variance, mean) for (mean, variance) in args]
            return funcs
        elif desc['type'] == 'uniform':
            def get_uniform(low, high):
                return lambda t: np.random.uniform(low=low, high=high, size=1)[0]
            min_amplitudes = parse_range(desc['min_amplitude'])
            max_amplitudes = parse_range(desc['max_amplitude'])
            args = itertools.product(min_amplitudes, max_amplitudes)
            funcs = [get_uniform(low, high) for (low, high) in args]
            return funcs
        elif desc['type'] == 'simple_predict_error':
            def simple_predict_error(mean_amp, var_amp, dur_samps_amp, samp_rate):
                MIN_ANOMALY_VALUES = 10000
                dur_between_amp_anomaly = np.floor(np.random.power(1, size=MIN_ANOMALY_VALUES)*dur_samps_amp)
                amp_anomaly_points = incremental_sum(dur_between_amp_anomaly)

                # NOTE still might be some issues with zeros, but these should mostly be surpressed implicitely
                # FIXME [ ]

                # This would suggest we should have multiple anomalies per interval
                # thus we ignore this and fill all samples with an anomaly
                if dur_samps_amp <= 1.0:
                    anomaly_values = np.zeros(MIN_ANOMALY_VALUES)
                    for i in range(0, MIN_ANOMALY_VALUES):
                        anomaly_values[i] = np.random.normal(loc=mean_amp, scale=var_amp)
                # Normal operation with potentially more than one anomaly per interval (which we ignore)
                else:
                    anomaly_values = np.zeros(int(amp_anomaly_points[-1]) + 1)
                    for point in amp_anomaly_points:
                        anomaly_values[int(point)] = np.random.normal(loc=mean_amp, scale=var_amp)

                #print(anomaly_values[0:100])
                def inner_function(t):
                    # due to inexact floating point, this should be even
                    # but we round to the nearest because it might not be
                    # so ceiling and floor would not be appropriate
                    index = int(np.around(t/sample_rate))
                    amp_distortion = anomaly_values[index]
                    return amp_distortion

                return inner_function
            mean_amplitudes = parse_range(desc['mean_amplitude'])
            var_amplitudes = parse_range(desc['var_amplitude'])
            # we do not take into account more than one error happening
            # during a sample period (one error every sample period is the
            # furthest we can go)
            dur_samples_amplitude = parse_range(desc['dur_samples_amplitude'])
            sample_rate = float(desc['sample_rate'])
            args = itertools.product(
                mean_amplitudes,
                var_amplitudes,
                dur_samples_amplitude,
                [sample_rate]
            )
            funcs = [simple_predict_error(
                m_amps, v_amps, d_s_amp,
                sr
            ) for (m_amps, v_amps, d_s_amp, sr) in args]
            return funcs
        elif desc['type'] == 'sine_with_phase_error':
            periods = parse_range(desc['period'])
            amplitudes = parse_range(desc['amplitude'])
            starting_phases = parse_range(desc['phase'])

            mean_phases = parse_range(desc['mean_phase'])
            var_phases = parse_range(desc['var_phase'])
            dur_samples_phase = parse_range(desc['dur_samples_phase'])
            sample_rate = parse_range(desc['sample_rate'])
            args = itertools.product(
                periods,
                amplitudes,
                starting_phases,
                mean_phases,
                var_phases,
                dur_samples_phase,
                sample_rate,
            )
            funcs = [sine_with_phase_error(
                period, amp, starting_phase,
                m_phases, v_phases, d_s_phase, sr
            ) for (period, amp, starting_phase, m_phases, v_phases, d_s_phase, sr) in args]
            return funcs
        elif desc['type'] == 'phase_error':
            def phase_error(period, amplitude, starting_phase,
                            mean_phase, var_phase, dur_samps_phase, sample_rate):
                predicted_sine = sine_with_phase_error(period, amplitude, starting_phase,
                                                       mean_phase, var_phase, dur_samps_phase, sample_rate)
                original_sine = sine(period, amplitude, starting_phase)

                def inner_function(t):
                    return original_sine(t) - predicted_sine(t)

                return inner_function

            periods = parse_range(desc['period'])
            amplitudes = parse_range(desc['amplitude'])
            starting_phases = parse_range(desc['phase'])

            mean_phases = parse_range(desc['mean_phase'])
            var_phases = parse_range(desc['var_phase'])
            dur_samples_phase = parse_range(desc['dur_samples_phase'])
            sample_rate = parse_range(desc['sample_rate'])
            args = itertools.product(
                periods,
                amplitudes,
                starting_phases,
                mean_phases,
                var_phases,
                dur_samples_phase,
                sample_rate,
            )
            funcs = [phase_error(
                period, amp, starting_phase,
                m_phases, v_phases, d_s_phase, sr
            ) for (period, amp, starting_phase, m_phases, v_phases, d_s_phase, sr) in args]
            return funcs
        else:
            print("ERROR: BAD DESCRIPTOR")
            exit(-1)

    global_funcs = Parallel(n_jobs=cpu_count(), verbose=10)(
        delayed(parse_descriptor)(desc) for desc in descriptors)

    def summer(func1, func2):
        def _sum(t):
            return func1(t) + func2(t)
        return _sum

    res = global_funcs[0]
    for i in range(0, len(global_funcs) - 1):
        temp = itertools.product(res, global_funcs[i+1])
        res = [summer(func1, func2) for (func1, func2) in temp]

    return res

def parse_pre_noise(descriptors, sample_rate):
    '''
    Should produce a list of functions to be
    called where each function returns the generated pre-noise.

    i.e. calling the function returns an array of all the generated points

    flare_star() -> [entire flare array]
    '''

    global_funcs = list()

    def parse_descriptor(desc):
        if desc['type'] == 'flare-sxr':
            A = parse_range(desc['A'])
            B = parse_range(desc['B'])
            C = parse_range(desc['C'])
            D = parse_range(desc['D'])
            E = parse_range(desc['E'])
            F = parse_range(desc['F'])
            args = itertools.product(A, B, C, D, E, F)
            funcs = [get_flare_star(A, B, C, D, E, F) for (A, B, C, D, E, F) in args]
            return funcs
        elif desc['type'] == 'flare-gj1234':
            amplitude = parse_range(desc['amplitude'])
            tot_time = parse_range(desc['tot_time'])
            args = itertools.product(amplitude, tot_time)
            funcs = [get_flare_star_v2(amplitude, tot_time, sample_rate)
                     for (amplitude, tot_time) in args]
            return funcs
        elif desc['type'] == 'skip':
            return [lambda: []]
        else:
            print("ERROR: BAD DESCRIPTOR")
            exit(-1)

    global_funcs = Parallel(n_jobs=cpu_count(), verbose=10)(
        delayed(parse_descriptor)(desc) for desc in descriptors)

    return list(itertools.product(*global_funcs))

def _gen(desc_file, config_name, output_dir, plot, new_data_fmt):
    utils.DEBUG = False

    sample_rate = desc_file['signal']['sample_rate']
    # NOTE make start and end length in terms of points and not seconds
    start_len = desc_file['signal']['start_len'] * sample_rate
    end_len = desc_file['signal']['end_len'] * sample_rate

    dcs = parse_range(desc_file['signal']['dc'])

    funcs = parse_noise_descriptors(desc_file['noise'])
    pre_noise_func_groups = parse_pre_noise(desc_file['pre-noise'], sample_rate)

    def gen_template(u0, te):
        # width = int(math.ceil(te/sample_rate))
        width = int(math.ceil(te/2.0))
        # makes sure that both sides are roughly the same DC for shifting star gen stuff later
        template = [utils.nfd_pzlcw(u0, 0, te, t) for t in range(-width, width+sample_rate,
                                                                 sample_rate)]
        # 2.5, log10 from NFD paper (I think) TODO check
        template = 2.5*np.log(template)

        # NOTE: makes sure that all templates start at 0 DC
        template -= np.min(template)
        return template

    u0s = parse_range(desc_file['signal']['u0'])
    tes = parse_range(desc_file['signal']['te'])
    args = itertools.product(u0s, tes)
    templates = Parallel(n_jobs=cpu_count(), verbose=10)(delayed(gen_template)(u0, te) for (u0, te) in args)

    tot_iter = len(funcs)*len(templates)*len(dcs)*len(pre_noise_func_groups)
    print('Amount to gen: {}'.format(tot_iter))

    with Manager() as manager:
        StarCount = namedtuple('StarCount', ['value', 'lock'])
        star_count = StarCount(manager.Value('i', 0),
                               manager.Lock())

        # NOTE optimal write speed seems to be setting to number of threads
        #      used which makes intuitive sense as a thread finishes one db opens up
        new_data_writer = NewDataWriter(output_dir, manager.Lock, 16, desc_file) if new_data_fmt else None

        def gen_signal(f, pre_noise_funcs, template, dc):
            i = 0

            ran = list(range(i, i + start_len, sample_rate))
            start = [f(t) for t in ran]
            # NOTE: plus sample_rate on all to get the next point (not the last point used)
            i = ran[-1] + sample_rate

            # XXX TODO Actually determine the best logarithmic way of doing things
            pre_noise_without_signal = []
            pre_noise = []
            for pnf in pre_noise_funcs:
                data = 2.5*np.log(np.array(pnf()) + 1.0)
                if len(data) > 0:
                    data = data - np.min(data)
                    pre_noise += list(data)
                #pre_noise += pnf()

            if len(pre_noise) > 0:
                #raise Exception("greater than 0: {}".format(pre_noise))
                ran = list(range(i, i + len(pre_noise)*sample_rate, sample_rate))
                pre_noise_without_signal = [f(t) for t in ran]
                i = ran[-1] + sample_rate
                pre_noise = \
                    [pre_noise_without_signal[i] + pre_noise[i] for i in range(0, len(pre_noise))]

            ran = list(range(i, i + len(template)*sample_rate, sample_rate))
            middle_without_signal = [f(t) for t in ran]
            middle = [middle_without_signal[i] + template[i] for i in range(0, len(template))]
            i = ran[-1] + sample_rate

            ran = list(range(i, i + end_len, sample_rate))
            end = [f(t) for t in ran]
            i = ran[-1] + sample_rate

            spow = signal_power(template, sample_rate)
            # NOTE: only considers the noise power in the part with the microlensing signal
            npow = noise_power(middle_without_signal, sample_rate)

            start = np.array(start)
            # NOTE: fixed by making templates start at 0 DC
            #       - visually inspected to have not discontinuities
            #start += template[0]
            start = list(start)

            end = np.array(end)
            # NOTE: fixed by making templates start at 0 DC
            #       - visually inspected to have not discontinuities
            #end += template[-1]
            end = list(end)

            # [ ] TODO make it where we also store the signals peak point (very easy to do)
            #     will make updating code here not have to change match filter code
            # [ ] TODO fix it also in match filter
            datum = np.array(start+pre_noise+middle+end)
            #datum = -1.0*np.array(start+pre_noise+middle+end)
            datum = {
                'samples': datum + dc,
                'spow': spow,
                'npow': npow,
                'snr': spow - npow,
                'tl': len(start) + len(pre_noise),
                'tr': len(start) + len(pre_noise) + len(middle)
            }

            write_star(datum, config_name, output_dir, plot, star_count, new_data_writer)

        Parallel(n_jobs=cpu_count(), verbose=10)(
            delayed(gen_signal)(f, pnfs, t, dc) for f, pnfs, t, dc in itertools.product(
            funcs, pre_noise_func_groups, templates, dcs))

        if new_data_writer:
            new_data_writer.close()

if __name__ == '__main__':
    main()
