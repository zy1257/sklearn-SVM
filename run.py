import jieba
import pickle
from collections import Counter
from math import log
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
'''
可调整参数
most_common = 3000，选取较为常见的特征
num = 950, 训练使用数据数量
C = 3.0

'''
def stop_build():
    stop = {}
    with open('.\\中文停用词表.txt', encoding = 'utf-8') as f:
        n = 0
        for i in f:
            stop[i.strip()] = n
            n += 1
        stop[' ' ] = -1
        stop['\n'] = -2
    with open('.\\中文停用词表.pkl', 'wb') as f:
        pickle.dump(stop, f)

with open('.\\中文停用词表.pkl', 'rb') as f:
    stop = pickle.load(f)

def D_n_build(num,stop, most_common=3000):  # 特征项出现的文档数
    D_n = {}
    for k in ('neg','pos'):
        for j in range(num):
            file = '.\\2000\\{}\\{}.{}.txt'.format(k,k,j)
            with open(file, encoding='utf-8') as f:
                f_jieba = jieba.cut(f.read())
                temp = []
                for i in f_jieba:
                    if i not in temp:
                        temp.append(i)
                for i in temp:
                    if i not in stop and i not in D_n:
                        D_n[i] = 1
                    elif i in D_n:
                        D_n[i] += 1
    D_n = dict(Counter(D_n).most_common(most_common))
    return D_n

def D_build(D_n,stop, num = 1000, num_pass=0):  # 生成特征项权重，给特征编号
    D, s = {},[]
    for k in ('neg','pos'):
        for j in range(num):
            file = '.\\2000\\{}\\{}.{}.txt'.format(k,k,j+num_pass)
            with open(file, encoding='utf-8') as f:
                f_jieba = jieba.cut(f.read())
                for i in f_jieba:
                    if i not in stop and i in D_n:
                        s.append(i)
            s_len = len(s)
            s_dic = dict(Counter(s))
            s_lis = list(s_dic)
            for i in s_lis:
                if i in D_n:
                    s_dic[i] = s_dic[i]/s_len * log(2*1000/D_n[i],10)
            D['{}{}'.format(k,j)] = s_dic
            s = []
    n_rank = 0
    D_n_rank = D_n.copy()
    for key in D_n_rank:
        D_n_rank[key] = n_rank
        n_rank += 1
    for k in ('neg','pos'):
        for j in range(num):
            np_num = '{}{}'.format(k,j)
            t_keys = list(D[np_num].keys())
            for key in t_keys:
                D[np_num].update({D_n_rank[key]:D[np_num].pop(key)})
    return D

def train_data(D, num):
    x_train, y_train = [],[]
    n_p = {'neg':-1,'pos':1}
    for k in ('neg','pos'):
        for j in range(num):
            base = np.zeros(3000)
            np_num = '{}{}'.format(k,j)
            t_keys = list(D[np_num].keys())
            for key in t_keys:
                base[key] = D[np_num][key]
            x_train.append(list(base))
            y_train.append(n_p[k])
    return x_train, y_train

def run(D_n, num=950):
    D = D_build(D_n,stop, num)
    x_train, y_train = train_data(D, num)
    D_test = D_build(D_n,stop,1000-num,num)
    x_test, y_test = train_data(D_test,1000-num)
    clf = svm.SVC(C=2.0, kernel = 'linear')
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    y_test = np.array(y_test)
    acc = y_test==y_predict
    acc = acc.sum()/len(acc)
    joblib.dump(clf,'emotion_svm.model')
    print(acc)
    
def demo(D_n, D_n_rank,content):
    clf = joblib.load('emotion_svm.model')
    s_jieba = jieba.cut(content)
    s = []
    for i in s_jieba:
        if i in D_n:
            s.append(i)
    s_len = len(s)
    s_dic = dict(Counter(s))
    s_lis = list(s_dic)
    for i in s_lis:
        s_dic[i] = s_dic[i]/s_len * log(2*1000/D_n[i],10)
    t_keys = list(s_dic.keys())
    for key in t_keys:
        s_dic.update({D_n_rank[key]:s_dic.pop(key)})
    base = np.zeros(3000)
    t_keys = list(s_dic.keys())
    for key in t_keys:
        base[key] = s_dic[key]
    x_cont = list(base)
    x_cont = np.array(x_cont).reshape(1,-1)
    res = clf.predict(x_cont)
    if res == 1:
        return 'Good'
    else:
        return 'Bad'

D_n = D_n_build(1000, stop)
n_rank = 0
D_n_rank = D_n.copy()
for key in D_n_rank:
    D_n_rank[key] = n_rank
    n_rank += 1
run(D_n, num=950)
while 1:
    content = input('请输入评语:\n')
    if content == '':
        break
    print(demo(D_n, D_n_rank,content))
