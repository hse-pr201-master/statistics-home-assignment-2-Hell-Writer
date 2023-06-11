import numpy as np
from scipy import stats as sts
from functools import reduce
from collections import Counter

def mlf1(n, i=10):
    '''Функция правдоподобия из 1 задачи'''
    left = (i-1)/n
    right = reduce(lambda x,y: x*y, [1 - j/n for j in range(i)])
    return left * right


def evf1(n):
    '''Функция матожидания из 1 задачи'''
    sum_list = []
    for j in range(2,n+2):
        left = (j-1)/n
        right = reduce(lambda x,y: x*y, [1-(i/n) for i in range(j-1)])
        val = left*right*j
        sum_list.append(val)
    return sum(sum_list)


def t_bootstrap(arr, alpha, n_bootstraps):
    '''Вычисляет критические значения бутстрепа матожидания выборки'''
    right_percentile = 1-alpha/2
    left_percentile = alpha/2
    size = len(arr)
    avg = sum(arr)/size
    stdev = np.sqrt(np.var(arr))
    tstats = []
    bootstraped_data = np.random.choice(arr, size*n_bootstraps)
    for i in range(n_bootstraps):
        avgest = sum(bootstraped_data[size*i:size*(i+1)])/size
        stderr = np.sqrt(np.var(arr))
        val = (avgest-avg)/stderr
        tstats.append(val)
    tstats = sorted(tstats)
    rcrit = avg - tstats[int(round(left_percentile*n_bootstraps))]*stdev
    lcrit = avg - tstats[int(round(right_percentile*n_bootstraps))]*stdev
    return (lcrit, rcrit)


def t_bootstrap_2_pvalue(arr1, arr2, n_bootstraps):
    '''Вычисляет pvalue бутстрепа матожидания разницы двух выборок'''
    size1 = len(arr1)
    size2 = len(arr2)
    avg = sum(arr1)/size1 - sum(arr2)/size2
    stdev = np.sqrt((np.var(arr1) + np.var(arr2))*(1/size1 +1/size2))
    tstats = []
    bootstraped_data_1 = np.random.choice(arr1, size1*n_bootstraps)
    bootstraped_data_2 = np.random.choice(arr2, size2*n_bootstraps)
    for i in range(n_bootstraps):
        avgest = sum(bootstraped_data_1[size1*i:size1*(i+1)])/size1 - sum(bootstraped_data_2[size2*i:size2*(i+1)])/size2
        stderr = stdev
        val = (avgest-avg)/stderr
        tstats.append(val)
    tstats = sorted(tstats)
    for i in range(len(tstats)):
        rpval = 1 - i/len(tstats)
        if tstats[i] > avg/stderr:
            break
    return 2*rpval 


def means_dif(sample1, sample2, axis=None):
    if axis:
        return np.mean(sample1, axis=axis) - np.mean(sample2, axis=axis)
    else:
        return (sum(sample1)/len(sample1)) - (sum(sample2)/len(sample2))


def od_ratio(sample):
    cnt = Counter(sample)
    return (cnt[0]*cnt[3])/(cnt[1]*cnt[2])


def ci_maker_kd(dist, sigma, alpha):
    '''Доверительный интервал для
    матожидания в случае известной дисперсии'''
    avg = dist.mean()
    n = len(dist)
    z = sts.norm.interval(1-alpha)[1]
    lcrit = avg - sigma*z/np.sqrt(n)
    rcrit = avg + sigma*z/np.sqrt(n)
    return [lcrit, rcrit]


def ci_maker_ud(dist, alpha):
    '''Доверительный интервал для
    матожидания в случае неизвестной дисперсии'''
    avg = dist.mean()
    n = len(dist)
    sigma = dist.std()
    t = sts.t.interval(1-alpha, df=n-1)[1]
    lcrit = avg - sigma*t/np.sqrt(n)
    rcrit = avg + sigma*t/np.sqrt(n)
    return [lcrit, rcrit]
