import scipy as sc
from scipy.stats import truncnorm
import numpy as np



def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


# 求截断正态分布下左侧的积分值占所有积分值的比例
def get_truncated_normal_probility(mean=0, sd=1, low=0, upp=10, value=1):
    if value <= low:
        return 0
    if value >= upp:
        return 1
    denominator = sc.stats.norm.cdf(upp, mean, sd * sd) - sc.stats.norm.cdf(low, mean, sd * sd)
    numerator = sc.stats.norm.cdf(value, mean, sd * sd) - sc.stats.norm.cdf(low, mean, sd * sd)
    return numerator / denominator


def Mzipf(a: np.float64, plateau: np.float64, min: np.uint64, max: np.uint64, size=None):
    """
    Generate Zipf-like random variables,
    but in inclusive [min...max] interval
    """
    if min == 0:
        raise ZeroDivisionError("")

    v = np.arange(min, max + 1)  # values to sample
    np.add(v, plateau)
    p = 1.0 / np.power(v, a)  # probabilities
    p /= np.sum(p)  # normalized

    return np.random.choice(v, size=size, replace=True, p=p)


def Mzipf_pops(a: np.float64, plateau: np.float64, min: np.uint64, max: np.uint64, size=None):
    if min == 0:
        raise ZeroDivisionError("")

    v = np.arange(min, max + 1)  # values to sample
    np.add(v, plateau)
    p = 1.0 / np.power(v, a)  # probabilities
    p /= np.sum(p)  # normalized
    return p


def reset_size(s, e, index, origin_size, final_size, tp: 0):
    if tp == 0:
        return int((index - s) / (e - s) * (final_size - origin_size) + origin_size)
