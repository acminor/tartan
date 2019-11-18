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
@click.option('-s', '--sample-rate',
              type=int,
              required=True)
@click.option('--desc-file', type=click.Path(), required=True)
@click.option('-TE-min', type=float, required=True)
@click.option('-TE-max', type=float, required=True)
@click.option('-TE-density', type=int, required=True)
@click.option('-plot', type=bool, default=False)
def main(output_dir, sample_rate, desc_file,
         te_min, te_max, te_density, plot):
    config = Path(desc_file).stem

    a = open(desc_file)
    desc_file = toml.load(a)
    stars = _gen(sample_rate, te_min, te_max, te_density, desc_file)

    if not plot and len(list(Path(output_dir).glob('*'))) != 0:
        if click.confirm('Output dir is not empty. Empty?'):
            for f in Path(output_dir).glob('*'):
                f.unlink()

    for i, star in enumerate(stars):
        path_name = (
            "config={},".format(config) +
            "len={},".format(len(star['samples'])) +
            "spow={:.2f},".format(star['spow']) +
            "npow={:.2f},".format(star['npow']) +
            "snr={:.2f}".format(star['snr'])
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

            with open(str(Path(output_dir)/Path('star' + str(i) + '.mpk')), 'wb+') as file:
                file.write(msgpack.packb(list(star['samples'])))

            star['samples'] = 'star' + str(i) + '.mpk'

            with open(str(Path(output_dir)/Path('star' + str(i) + '.toml')), 'w+') as file:
                toml.dump(star, file)

def sine(period, amplitude, phase):
    return lambda t: amplitude*math.sin((2*math.pi*(t/period) + np.radians(phase)) % (2*math.pi))

def cosine(period, amplitude, phase):
    return lambda t: amplitude*math.cos((2*math.pi*(t/period) + np.radians(phase)) % (2*math.pi))

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

    global_funcs = Parallel(n_jobs=8, verbose=10)(delayed(parse_descriptor)(desc) for desc in descriptors)
    #for desc in descriptors:
    #    global_funcs.append(parse_descriptor(desc))

    def summer(func1, func2):
        def _sum(t):
            return func1(t) + func2(t)
        return _sum

    res = global_funcs[0]
    for i in range(0, len(global_funcs) - 1):
        temp = itertools.product(res, global_funcs[i+1])
        res = [summer(func1, func2) for (func1, func2) in temp]

    return res

def _gen(sample_rate, te_min, te_max, te_density, desc_file):
    utils.DEBUG = False

    funcs = parse_noise_descriptors(desc_file['noise'])

    start_len = desc_file['signal']['start_len']
    end_len = desc_file['signal']['end_len']
    sample_rate = desc_file['signal']['sample_rate']

    def gen_template(u0, te):
        width = int(te/sample_rate)
        template = [utils.nfd_pzlcw(u0, 0, te, t) for t in range(-width, width, sample_rate)]
        # 2.5, log10 from NFD paper (I think) TODO check
        template = 2.5*np.log(template)
        return template

    u0s = parse_range(desc_file['signal']['u0'])
    tes = parse_range(desc_file['signal']['te'])
    args = itertools.product(u0s, tes)
    templates = Parallel(n_jobs=8, verbose=10)(delayed(gen_template)(u0, te) for (u0, te) in args)

    tot_iter = len(funcs)*len(templates)
    print('Amount to gen: {}'.format(tot_iter))


    def gen_signal(f, template):
        start = [f(t) for t in range(0, start_len, sample_rate)]

        middle_without_signal = [f((i + len(start))*sample_rate) for i in range(0, len(template))]
        middle = [f((i + len(start))*sample_rate) + template[i] for i in range(0, len(template))]

        end_start = sample_rate*(len(start)+len(middle))
        end = [f(t) for t in range(end_start, end_start + end_len, sample_rate)]

        spow = signal_power(template, sample_rate)
        npow = noise_power(start+middle_without_signal+end, sample_rate)

        start = np.array(start)
        start += template[0]
        start = list(start)

        end = np.array(end)
        end += template[-1]
        end = list(end)

        datum = np.array(start+middle+end)
        datum = {
            'samples': datum,
            'spow': spow,
            'npow': npow,
            'snr': spow - npow
        }

        return datum

    data = Parallel(n_jobs=8, verbose=10)(
        delayed(gen_signal)(f, t) for f, t in itertools.product(funcs, templates))
#    data = list()
#    for template in templates:
#        for f in funcs:
#            data.append(gen_signal(f, t))

    return data

if __name__ == '__main__':
    main()
