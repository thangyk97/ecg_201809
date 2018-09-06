import numpy as np
from biosppy.signals import ecg
#
from hrv.classical import time_domain
from hrv.classical import frequency_domain
from hrv.classical import non_linear


def cal_r_peaks(signal, sampling_rate):
    rpeaks, = ecg.hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)
    rpeaks, = ecg.correct_rpeaks(signal=signal, rpeaks=rpeaks, sampling_rate=sampling_rate, tol=0.05)
    templates, rpeaks = ecg.extract_heartbeats(signal=signal, rpeaks=rpeaks, sampling_rate=sampling_rate, before=0.2, after=0.4)
    return rpeaks

def prepare_test_data():
    data2 = np.loadtxt("eval.txt")

    eval_data = []
    eval_label = []
    for array in data2:
        eval_label.append(int (array[10800]))
        array = array[:-1]
        eval_data.append(array)
    print (np.shape(eval_data))
    print (np.shape(eval_label))
    result = []
    for x in eval_data:
        rpeaks = cal_r_peaks(x, 360)
        tmp = []
        rri = []
        for i in range(len(rpeaks)-1):
            tmp.append(rpeaks[i + 1] - rpeaks[i])
        rri = np.divide(tmp, 360)
        time = time_domain(rri)
        fre = frequency_domain(
            rri=rri,
            fs=9.0,
            method='welch',
            interp_method='cubic',
            detrend='linear'
        )
        non = non_linear(rri)
        sd1_sd2 = np.divide(non['sd1'], non['sd2'])
        tmpResult = [time['mhr'], time['mrri'], time['nn50'], time['pnn50'], time['rmssd'], time['sdnn'], fre['lf_hf'],sd1_sd2 ]
        result.append(tmpResult)

    for j in range(len(result)):
        result[j].append(eval_label[j])
    print(np.shape(result))
    np.save("test_data", result)

def prepare_training_data():
    data1 = np.loadtxt("training.txt")    
    
    training_data = []
    training_label = []
    for array in data1:
        training_label.append(int(array[10800]))
        array = array[:-1]
        training_data.append(array)

    print (np.shape(training_data))
    print (np.shape(training_label))
    result = []
    for x in training_data:
        rpeaks = cal_r_peaks(x, 360)
        tmp = []
        rri = []
        for i in range(len(rpeaks)-1):
            tmp.append(rpeaks[i + 1] - rpeaks[i])
        rri = np.divide(tmp, 360)
        time = time_domain(rri)
        fre = frequency_domain(
            rri=rri,
            fs=9.0,
            method='welch',
            interp_method='cubic',
            detrend='linear'
        )
        non = non_linear(rri)
        sd1_sd2 = np.divide(non['sd1'], non['sd2'])
        tmpResult = [time['mhr'], time['mrri'], time['nn50'], time['pnn50'], time['rmssd'], time['sdnn'], fre['lf_hf'],sd1_sd2 ]
        result.append(tmpResult)

    for j in range(len(result)):
        result[j].append(training_label[j])
    print(np.shape(result))
    np.save("training_data", result)

def main(unused_argv):
    prepare_test_data()
    prepare_training_data()

if __name__ == "__main__":
    tf.app.run()