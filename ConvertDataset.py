from scipy.io import wavfile
from scipy.fftpack import dct
import numpy as np
import numpy.fft as fft
from os import makedirs, walk
from skimage.io import imsave, imread
from skimage.exposure import rescale_intensity
from skimage import img_as_uint
import matplotlib.pyplot as plt

def get_frames(data, sr, wlen=0.025, wstep=0.010, debug=False):
    if wstep > wlen:
        raise Exception("Шаг фрейма больше его длины!")
    flen = round(wlen * sr)
    fstep = round(wstep * sr)

    overflownum = wlen // wstep
    fnum = int(np.ceil(len(data) / fstep))

    frames = np.ndarray((fnum - 1, flen))
    if debug:
        print(flen, fstep, overflownum, fnum, data.shape)
    for i in range(int(fnum - overflownum - 1)):
        fstart = i * fstep
        frames[i] = data[fstart:fstart+flen]
    last = int(fnum - overflownum)
    last_start = last * fstep
    last_frame = data[last_start:]
    last_frame = np.append(last_frame, np.zeros(flen - len(last_frame)))
    frames[last] = last_frame
    
    return frames

def get_spectrogram(data, noise_cutoff=8):
    spec = data
    spec /= spec.max() # norm
    with np.errstate(divide='ignore'): 
        spec = np.log10(spec)
    spec[spec < -noise_cutoff] = -noise_cutoff

    return spec

def window_Hamming(wlen):
    filter = np.zeros(wlen)
    for n in range(wlen):
        filter[n] = 0.54 - 0.46 * np.cos((2 * np.pi * n) / (wlen - 1))
    return filter

def hzToMel(hz):
    return 2595 * np.log10(1 + hz / 700.0)

def melToHz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)

def getFilters(n=32, fsize=276*2, sr=22050):
    lowend = 0
    highend = sr / 2

    lowmel = hzToMel(lowend)
    highmel = hzToMel(highend)
    melspace = np.linspace(lowmel, highmel, n+2)

    bins = np.floor((fsize+1) * melToHz(melspace) / sr)

    filters = np.zeros([n, fsize // 2])
    for j in range(0, n):
        for i in range(int(bins[j]), int(bins[j + 1])):
            filters[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            filters[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
    return filters

def convertFile(path, debug=False):
    sr, data = wavfile.read(path)
    duration = 30 - 0.025
    sample_num = int(sr*duration)
    data = data[:sample_num]
    if debug:
        print(sr, sample_num, data.shape)
    if sample_num > len(data):
        print(f"Файл по адресу {path} недостаточной длины, пропускается")
        return None, None


    preemp = 0.66
    preemp_data = np.append(data[0], data[1:] - preemp * data[:-1])
    if debug:
        print("PDS:", preemp_data.shape)
    framed_data = get_frames(preemp_data, sr, wlen=0.025, wstep=0.010, debug=debug)
    flen = framed_data.shape[1]
    filter = window_Hamming(flen)
    windowed_data = framed_data * filter

    specter = np.abs(fft.rfft(windowed_data))
    spectrogram = get_spectrogram(specter)
    filterMatrix = getFilters()
    normFilterMatrix = filterMatrix.T / filterMatrix.sum(axis=1)
    melSpectrogram = np.transpose(normFilterMatrix).dot(np.transpose(spectrogram))
    numToKeep = 16
    mfcc = dct(melSpectrogram.T, type=2, axis=1, norm='ortho')[:, : numToKeep]

    return melSpectrogram, mfcc

if __name__ == "__main__":
    DEBUG = False
    COMPARE_WITH_DEMO = False

    rootPath = "./Dataset/"
    msPath = rootPath + "Mel Spectrograms/"
    mfccPath = rootPath + "MFCCs/"
    print(f"Creating folders at {msPath} and {mfccPath}")
    makedirs(msPath, exist_ok=True)
    makedirs(mfccPath, exist_ok=True)

    genresPath = rootPath + "genres/"
    print(f"Looking at genres at {genresPath}")

    for root, dirs, files in walk(genresPath):
        print(f"Looking in directory: {root}")
        makedirs(msPath + root.split('/')[-1], exist_ok=True)
        makedirs(mfccPath + root.split('/')[-1], exist_ok=True)
        for file in files:
            if DEBUG:
                print(f"  working on file: {root + '/' + file}")
            # файл jazz.00054.wav не открывается - испорчен, удаляем из датасета
            # 18 файлов несколько короче 30 секунд и пропускаются, однако можно уменьшить
            # требуемую длительнотсь на пару фреймов
            # короче 29.975 секунд только hiphop.00032.wav, пропускаем
            melSpec, mfcc = convertFile(root + '/' + file, debug=DEBUG)
            if melSpec is None or mfcc is None:
                continue
            melPath = msPath + root.split('/')[-1] + '/' + file.rstrip(".wav").replace(".", "_") + ".png"
            mfccsPath = mfccPath + root.split('/')[-1] + '/' + file.rstrip(".wav").replace(".", "_") + ".png"
            if DEBUG:
                print(f"Saving to {melPath} and {mfccsPath}")
            melSpec = rescale_intensity(melSpec, out_range="float")
            melSpec = img_as_uint(melSpec)
            imsave(melPath, melSpec)
            if COMPARE_WITH_DEMO:
                if melPath.endswith("jazz_00005.png"):
                    print("Found demo example")
                    im = imread(melPath)
                    fig, ax = plt.subplots(1, 1, figsize=(20, 6))
                    cax = ax.matshow(im.astype("float32"), interpolation="nearest", aspect="auto", cmap=plt.cm.plasma, origin="lower")
                    fig.colorbar(cax)
                    plt.xlabel('Номер фрейма')
                    plt.ylabel('Номер мел-фильтра')
                    plt.title("мел-спектрограмма")
                    plt.show()
            mfcc = rescale_intensity(mfcc, out_range="float")
            mfcc = img_as_uint(mfcc)
            imsave(mfccsPath, mfcc.T)
            if COMPARE_WITH_DEMO:
                if mfccsPath.endswith("jazz_00005.png"):
                    print("Found demo example")
                    im = imread(mfccsPath).T
                    plt.imshow(im[0:50].T, interpolation='antialiased')
                    plt.xlabel('Номер фрейма')
                    plt.ylabel('MFCC')
                    plt.show()