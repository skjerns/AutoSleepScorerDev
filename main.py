# -*- coding: utf-8 -*-
"""
Spyder Editor

main script for training/classifying
"""

from preproc import pre_process
import chainer


train,test = pre_process()



#%%



print('firing up the classifiers')
idx = np.arange(len(freq)); np.random.shuffle(idx);

freq = [freq[x] for x in idx]
hypno = [hypno[x] for x in idx]


y = hypno[split:]
#np.random.shuffle(y)
print('ADA 100')
clf = AdaBoostClassifier(n_estimators=100, random_state=42)
clf = clf.fit(freq[0:split], hypno[0:split])
pred = clf.score(freq[split:], y)
np.random.shuffle(y)
shuff = clf.score(freq[split:], y)
print (['prediction score: ', pred])
print (['random expected: ', shuff])