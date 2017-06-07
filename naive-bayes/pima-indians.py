
## from http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
#
import csv

def load_csv(filename):
    lines = csv.reader(open(filename, 'r'))
    rows = []
    for r in enumerate(lines):
        line_content = r[1]
        rows.append([float(ri) for ri in line_content])
    return rows

fn = 'data/pima-indians.csv'
d = load_csv(fn)

import random
def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset)*split_ratio)
    trainset = dataset[:train_size]
    testset = dataset[train_size:]
    return trainset, testset

trs, tes = split_dataset(d, 0.7)
print("train set size: {0}, test set size: {1}".format(len(trs), len(tes)))

# print(d[0])
# a data row looks like: 6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0, 1.0
# with 8 input features and 1 class variable
# p(y|x) ∝ p(y,x)= p(y)*p(x|y)= p(y)*p(x1|y)*p(x2|y)*...*p(x8|y)
#           p(y) = count(y)/total
# suppose xi is a RV of Gaussian distribution, defined by mean(xi) and std_dev(xi)
#        p(x1|y) = count(x1,y)/count(y)

def sep_by_class(dataset):
    sep = { }
    for dr in dataset:
        c = dr[-1]
        if (c not in sep):
            sep[c] = []
        sep[c].append(dr)
    return sep

tr_sep = sep_by_class(trs)
for k in tr_sep.keys():
    print("{0} rows are of class {1}".format(len(tr_sep[k]), k))

import math
def calc_mean_stddev(dataset, i):
    s = sum([d[i] for d in dataset])
    mean = s/len(dataset)
    var = sum([(d[i]-mean)*(d[i]-mean) for d in dataset]) / (len(dataset)-1)
    stddev = math.sqrt(var)
    return mean, stddev

import scipy.stats as stats
def prob_xi(x, mean, stddev):
    return stats.norm.pdf(x, mean, stddev)*stddev

means = { }
stddevs = { }
prob_y = { }
for k in tr_sep.keys():
    means[k] = [ ]
    stddevs[k] = [ ]
    prob_y[k] = len(tr_sep[k]) / len(trs)
    for i in range(8):
        m, s = calc_mean_stddev(tr_sep[k], i)
        means[k].append(m)
        stddevs[k].append(s)
        print("[%d]: %.3f/%.3f"%(i, m, s))

true_count = 0
for td in tes:
    predict_prob = { }
    max_prob = float('-inf')
    max_key = float('-inf')
    for k in means.keys():
        p = [ ]
        pp = prob_y[k]
        for i in range(len(td)-1):
            m = means[k][i]
            s = stddevs[k][i]
            xi = td[i]
            prob = prob_xi(xi, m, s)
            p.append(prob)
            pp = pp*prob
        predict_prob[k] = pp
        if (pp > max_prob):
            max_prob = pp
            max_key = k
    correct = max_key == td[-1]
    c = 'F'
    if (correct):
        true_count = true_count+1
        c = 'T'
    print("%s: (p:%s/a:%s)\t\t%s"%(c, max_key, td[-1], predict_prob))
print("precision: %.3f"%(true_count/len(tes)))