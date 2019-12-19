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
from multiprocessing import cpu_count

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
@click.option('-o', '--output-dir',
              type=click.Path(),
              required=True)
@click.option('-i', '--desc-file', type=click.Path(), required=True)
@click.option('-plot', type=bool, default=False)
def main(output_dir, desc_file, plot):
    with open(desc_file) as a:
        desc_data = toml.load(a)

    if not plot and len(list(Path(output_dir).glob('*'))) != 0:
            if click.confirm('Output dir is not empty. Empty?'):
                for f in Path(output_dir).glob('*'):
                    f.unlink()

    config_name = Path(desc_file).stem
    stars = _gen(desc_data, config_name, output_dir, plot)

def write_star(star, config_name, output_dir, plot):
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
        plt.title(path_name)
        plt.show()
    else:
        star['id'] = path_name
        star['star_type'] = 'unknown'
        star['sample_rate'] = 15
        star['arima_model_file'] = ''

        # hopefully to prevent two files from being the same
        i = time()
        with open(str(Path(output_dir)/Path('star' + str(i) + '.mpk')), 'wb+') as file:
            file.write(msgpack.packb(list(star['samples'])))

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

def _gen(desc_file, config_name, output_dir, plot):
    utils.DEBUG = False

    sample_rate = desc_file['signal']['sample_rate']
    # NOTE make start and end length in terms of points and not seconds
    start_len = desc_file['signal']['start_len'] * sample_rate
    end_len = desc_file['signal']['end_len'] * sample_rate

    dcs = parse_range(desc_file['signal']['dc'])

    funcs = parse_noise_descriptors(desc_file['noise'])
    pre_noise_func_groups = parse_pre_noise(desc_file['pre-noise'], sample_rate)

    def gen_template(u0, te):
        width = int(math.ceil(te/sample_rate))
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

    def gen_signal(f, pre_noise_funcs, template, dc):
        i = 0

        ran = list(range(i, i + start_len, sample_rate))
        start = [f(t) for t in ran]
        # NOTE: plus sample_rate on all to get the next point (not the last point used)
        i = ran[-1] + sample_rate

        pre_noise_without_signal = []
        pre_noise = []
        for pnf in pre_noise_funcs:
            pre_noise += pnf()

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
        datum = {
            'samples': datum + dc,
            'spow': spow,
            'npow': npow,
            'snr': spow - npow,
            'tl': len(start) + len(pre_noise),
            'tr': len(start) + len(pre_noise) + len(middle)
        }

        write_star(datum, config_name, output_dir, plot)

    Parallel(n_jobs=cpu_count(), verbose=10)(
        delayed(gen_signal)(f, pnfs, t, dc) for f, pnfs, t, dc in itertools.product(
        funcs, pre_noise_func_groups, templates, dcs))

if __name__ == '__main__':
    main()
