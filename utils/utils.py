import scipy.signal as sig
import scipy.integrate as ing
import scipy.optimize as opti
import numpy as np
import matplotlib.backends as pltb
from collections import namedtuple
from copy import deepcopy
import math
import pyfftw

import matplotlib
matplotlib.use('Gtk3Agg')
import matplotlib.pyplot as plt

DEBUG = True

class Sampler:
    def __init__(self, sample_rate):
        self._sample_rate = sample_rate

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        self._sample_rate = sample_rate

    def freq_scale(self):
        return 1/self.sample_rate

    def sample(self, function, num_samples):
        return self.sample_from_to(function, -num_samples, num_samples)

    # NOTE abstract in terms of samples and not time
    def sample_from_to(self, function, begin, end):
        samples = list()
        for i in range(begin, end):
            samples.append(function(i*self.sample_rate))
        return np.array(samples)

def sinw(freq, t, phase=0, scale=1):
    return scale*math.sin((2.0*math.pi*freq*t + phase) % (2.0*math.pi))

def sqw(t, scale=1):
    return scale

def nfd_pzlcw(u0, t0, tE, t):
    u = u0 + abs((t-t0)/tE)
    mu = (u**2 + 2)/(math.sqrt(u**2 + 4)*u)
    return mu

# TODO validate
def classical_pzlcw(u0, t0, tE, t):
    u = math.sqrt(u0**2.0 + ((t-t0)/tE)**2.0)
    mu = (u**2.0 + 2.0)/(u*math.sqrt(u**2.0 + 4.0))
    return mu

def embed_in_gaussian_noise(signal, offset, length, scale=1):
    if length < signal.size:
        raise Exception('Noise should be longer than signal.')
    elif length < signal.size+offset:
        print('WARNING: OFFSET CAUSING EMBEDDED SIG. TRUNC.')

    noise = np.random.normal(size=length, scale=scale)
    output = deepcopy(noise)
    for i in range(offset, offset+signal.size):
        if offset+i > noise.size:
            break
        output[i] = signal[i-offset] + output[i]
    return output, noise

def embed_in_arb_noise(signal, offset, noise):
    if noise.size < signal.size:
        #raise Exception('Noise should be longer than signal.')
        print('WARNING: SIG. TRUNC.')
    elif noise.size < signal.size+offset:
        print('WARNING: OFFSET CAUSING EMBEDDED SIG. TRUNC.')

    output = deepcopy(noise)
    for i in range(offset, offset+signal.size):
        if offset+i >= noise.size:
            break
        output[i] = signal[i-offset] + output[i]
    return output, noise

def inner_product(template, signal, snf, freq_scale, template_pre_fft=False):
    if template.size > signal.size:
        size = template.size
    else:
        size = signal.size

    if not template_pre_fft:
        template = np.flip(template)
        #template_f = fft.fft(template, n=size)
        template = pyfftw.byte_align(template, n=pyfftw.simd_alignment)
        template_f = pyfftw.interfaces.numpy_fft.fft(template)
    else:
        template_f = template

    # signal_f = np.fft.fft(signal, n=size)

    signal = signal + 1000.0
    signal = pyfftw.byte_align(signal, n=pyfftw.simd_alignment)
    signal_f = pyfftw.interfaces.numpy_fft.fft(signal)
    signal_f = np.pad(signal_f, (0, template_f.size - signal_f.size), mode='constant')
    freqs = freq_scale * pyfftw.interfaces.numpy_fft.fftfreq(size)

    #@jit(nopython=True, cache=True)
    def blah(template_f, signal_f, snf, freqs):
        num = template_f*signal_f
        den = snf

        eq = num/den
        cond = freqs>=0

        eq2 = eq[:eq.size//2]
        eq2 = np.extract(cond, eq)
        freqs = np.extract(cond, freqs)

        #ip = 4*np.real(np.sum(eq2)*freq_scale/eq2.size)
        ip = 4*np.real(ing.trapz(eq2, x=freqs))
        #ip = 4*np.real(np.trapz(eq2, x=freqs))
        return ip, eq

    ip, eq = blah(template_f, signal_f, snf, freqs)
    #ip, eq = blah(template_f, signal_f, snf, None)

    if DEBUG:
        print('inner_product(): debug output')
        print('Freq. Scale: {}'.format(freq_scale))
        print('Inner Product: {}'.format(ip))
        print('Max over filter: {}'.format(np.max(eq)))

    return ip, eq

def easy_rfft(signal, length):
    signal = pyfftw.byte_align(signal, n=pyfftw.simd_alignment)
    signal_fft = pyfftw.interfaces.numpy_fft.fft(signal, n=length)

    # using the numpy fftfreq reference
    # [ ] TODO check if correct
    # - ie only concerned with pos. freq. in fft
    if (length % 2) == 1:
        # odd
        real_length = int((length - 1)/2)
    else:
        # even
        real_length = int((length / 2) - 1)

    signal_fft = signal_fft[0:real_length]

    return signal_fft

def inner_product_2(template, signal, fft_max_len):
    temp_fft = easy_rfft(template, fft_max_len)
    signal_fft = easy_rfft(signal, fft_max_len)

    left_res = temp_fft * np.conjugate(signal_fft)
    right_res = np.conjugate(temp_fft) * signal_fft

    res = np.real(np.sum(left_res + right_res))
    return res, None

def window_iter(data, offset, length):
    if data.size < offset+length and DEBUG:
        print('REDUCED LENGTH WINDOW ITER: TRUNC')

    for i in range(offset, offset+length):
        if i > data.size:
            return
        yield data[i]

def window_arr(data, offset, length):
    return np.array([e for e in window_iter(data, offset, length)])
