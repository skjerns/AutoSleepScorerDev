# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:10:40 2017

@author: Simon
"""
import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import re
import tools
cmap = sns.cubehelix_palette(8, start=2.8, rot=-.1, as_cmap=True)
fsize = np.array((4,3.5))

#%%

pkl = pickle.load(open('.\\results\\new_results_electrodes_cnn.pkl', 'rb'))
t_label = ['W', 'S1', 'S2', 'SWS', 'REM']
for key in pkl:
    confmat = [x[4] for x in pkl[key]]
    tools.plot_confusion_matrix ('recurrent_newest'+ key + '.png',np.mean(confmat,0), t_label,figsize=fsize, perc=True, cmap=cmap, title='')

pkl = pickle.load(open('.\\results\\results_recurrent_seqlen-rnn6.pkl', 'rb'))
#pkl['feat-LSTM'] = pkl.pop('pure_rnn_do_6')
confmat = [x[4] for x in pkl[list(pkl.keys())[0]]]
tools.plot_confusion_matrix ('recurrent_'+ key +'_newest.png',np.mean(confmat,0), t_label,figsize=fsize, perc=True, cmap=cmap, title='Feat-LSTM')


pkl = pickle.load(open('.\\1098_cnn3adam_filter_SS_cnnnobal_results.pkl', 'rb'))
t_label = ['W', 'S1', 'S2', 'SWS', 'REM']
for key in pkl:
    confmat = [x[4] for x in pkl[key]]
#    plt.close('all')
    tools.plot_confusion_matrix ('cropping'+ key + '.png',np.mean(confmat,0), t_label,figsize=fsize, perc=True,cmap=cmap, title='NewData {}'.format(key))


#%% Differenceplot

pkl = pickle.load(open('.\\results\\new_results_electrodes_cnn.pkl', 'rb'))
pkl.update(pickle.load(open('.\\results\\new_results_electrodes_feat.pkl', 'rb')))

t_label = ['W', 'S1', 'S2', 'SWS', 'REM']

cnn_eeg = np.mean([x[4] for x in pkl['cnn3l2 eeg']], 0)
cnn_eog = np.mean([x[4] for x in pkl['cnn3l2 eog']], 0)
cnn_emg = np.mean([x[4] for x in pkl['cnn3l2 emg']], 0)
cnn_all = np.mean([x[4] for x in pkl['cnn3l2 all']], 0)

ann_eeg = np.mean([x[4] for x in pkl['anneeg']], 0)
ann_eog = np.mean([x[4] for x in pkl['anneeg+eog']], 0)
ann_emg = np.mean([x[4] for x in pkl['anneeg+emg']], 0)
ann_all = np.mean([x[4] for x in pkl['annall']], 0)

tools.plot_difference_matrix('cnn-all-vs-eeg.png', cnn_eeg, cnn_all, t_label,figsize=fsize, perc=True, title='EEG+EMG+EOG minus EEG')
tools.plot_difference_matrix('cnn-all-vs-eog.png', cnn_eog, cnn_all, t_label,figsize=fsize, perc=True, title='EEG+EMG+EOG minus EEG+EOG')
tools.plot_difference_matrix('cnn-all-vs-emg.png', cnn_emg, cnn_all, t_label,figsize=fsize, perc=True, title='EEG+EMG+EOG minus EEG+EMG')

tools.plot_difference_matrix('ann-all-vs-eeg.png', ann_eeg, ann_all, t_label,figsize=fsize, perc=True, title='EEG+EMG+EOG minus EEG')
tools.plot_difference_matrix('ann-all-vs-eog.png', ann_eog, ann_all, t_label,figsize=fsize, perc=True, title='EEG+EMG+EOG minus EEG+EOG')
tools.plot_difference_matrix('ann-all-vs-emg.png', ann_emg, ann_all, t_label,figsize=fsize, perc=True, title='EEG+EMG+EOG minus EEG+EMG')


#%%
pkl = pickle.load(open('.\\results\\new_results_recurrent.pkl', 'rb'))
rec_ann = np.mean([x[4] for x in pkl['pure_rnn_do']], 0)
rec_cnn = np.mean([x[4] for x in pkl['CNN+LSTM_fc1']], 0)
tools.plot_confusion_matrix ('cnn+lstm.png', rec_cnn, t_label,figsize=fsize, perc=True, cmap=cmap, title='CNN+LSTM')
tools.plot_confusion_matrix ('feat-lstm.png', rec_ann, t_label,figsize=fsize, perc=True, cmap=cmap, title='Feat-LSTM')

tools.plot_difference_matrix('rec-ann-vs-cnn.png', rec_ann, rec_cnn , t_label,figsize=fsize, perc=True, title='feat-LSTM minus CNN+LSTM')

plt.close('all')

#tools.plot_difference_matrix('cnn-eeg-vs-eog.png', cnn_eeg, cnn_eog, target,figsize=fsize, perc=True, title='CNN EOG minus EEG')
#tools.plot_difference_matrix('cnn-eeg-vs-emg.png', cnn_eeg, cnn_emg, target,figsize=fsize, perc=True, title='CNN EMG minus EEG')
#tools.plot_difference_matrix('cnn-eeg-vs-all.png', cnn_eeg, cnn_all, target,figsize=fsize, perc=True, title='CNN all minus EEG')
#tools.plot_difference_matrix('ann-eeg-vs-eog.png', ann_eeg, ann_eog, target,figsize=fsize, perc=True, title='ANN EOG minus EEG')
#tools.plot_difference_matrix('ann-eeg-vs-emg.png', ann_eeg, ann_emg, target,figsize=fsize, perc=True, title='ANN EMG minus EEG')
#tools.plot_difference_matrix('ann-eeg-vs-all.png', ann_eeg, ann_all, target,figsize=fsize, perc=True, title='ANN all minus EEG')
#tools.plot_difference_matrix('cnn-vs-ann.png', ann_all, cnn_all, target,figsize=fsize, perc=True, title='CNN all minus ANN all')

#for key in pkl:
#    confmat = pkl[key]]
#    plt.close('all')
#    
#    tools.plot_confusion_matrix ('recurrent_'+ key + '.eps',np.mean(confmat,0), target, perc=True)
#%% Plots for presentation cnn

plt.figure(figsize=[8,3])
test_acc     = np.array([0.837,	0.850,	0.842,	0.848])
test_acc_min = np.array([0.820,	0.824,	0.811,	0.818])
test_acc_max = np.array([0.850,	0.872,	0.873,	0.869])


plt.plot(test_acc, 'bo')
plt.errorbar(np.arange(4),test_acc , [test_acc - test_acc_min, test_acc_max - test_acc],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test Accuracy')

plt.xticks(np.arange(4), ['EEG','EEG+EOG','EEG+EMG','All'])

test_f1     = np.array([0.710,	0.734,	0.732,	0.743])
test_f1_min = np.array([0.691,	0.706,	0.710,	0.721])
test_f1_max = np.array([0.736,	0.764,	0.761,	0.760])

plt.figure(figsize=[8,3])
plt.plot(test_f1, 'go')
plt.errorbar(np.arange(4),test_f1 , [test_f1 - test_f1_min, test_f1_max - test_f1],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test F1-score')

plt.xticks(np.arange(4), ['EEG','EEG+EOG','EEG+EMG','All'])

#%% Plots for presentation rnn
plt.figure(figsize=[5,3])
rec_acc     = np.array([0.848, 0.856])
rec_acc_min = np.array([0.818, 0.836])
rec_acc_max = np.array([0.869, 0.876])



plt.plot(rec_acc[0], 'bo')
plt.errorbar([0] ,rec_acc[:1] , [rec_acc[:1] - rec_acc_min[:1], rec_acc_max[:1] - rec_acc[:1]],
             fmt='.k', ecolor='gray', lw=1)
plt.plot(1,rec_acc[1], 'go')
plt.errorbar([1] ,rec_acc[1:] , [rec_acc[1:] - rec_acc_min[1:], rec_acc_max[1:] - rec_acc[1:]],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test Accuracy')
plt.xticks(np.arange(2), ['CNN','CNN+LSTM'])
plt.xlim([-1,2])

plt.figure(figsize=[5,3])
rec_f1     = np.array([0.743,0.764])
rec_f1_min = np.array([0.721,0.740])
rec_f1_max = np.array([0.760,0.785])

plt.plot(rec_f1[0], 'bo')
plt.errorbar(0,rec_f1[:1] , [rec_f1[:1] - rec_f1_min[:1], rec_f1_max[:1] - rec_f1[:1]],
             fmt='.k', ecolor='gray', lw=1)
plt.plot(1,rec_f1[1], 'go')
plt.errorbar(1,rec_f1[1:] , [rec_f1[1:] - rec_f1_min[1:], rec_f1_max[1:] - rec_f1[1:]],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test F1-score')
plt.xticks(np.arange(2), ['CNN','CNN+LSTM'])
plt.xlim([-1,2])

#%% plotting hand vs automatic
feat_acc     = np.array([0.812,  	0.834,  	0.836,  	0.847])
feat_acc_min = np.array([0.772,	0.805,	0.806,	0.823])
feat_acc_max = np.array([0.832,  	0.853,  	0.857,  	0.863  ])
feat_f1      = np.array([0.647,  	0.677,  	0.714 , 	0.730 ])
feat_f1_min  = np.array([0.617,	0.648,	0.687,	0.706])
feat_f1_max  = np.array([0.667,  	0.695,  	0.739,  	0.754])
rec_acc     = np.array([0.856])
rec_acc_min = np.array([0.836])
rec_acc_max = np.array([0.876])
rec_f1     = np.array([0.764])
rec_f1_min = np.array([0.740])
rec_f1_max = np.array([0.785])
frec_acc     = np.array([0.853])
frec_acc_min = np.array([0.815])
frec_acc_max = np.array([0.883])
frec_f1      = np.array([0.754])
frec_f1_min  = np.array([0.713])
frec_f1_max  = np.array([0.783])

plt.figure(figsize=[6,3])
plt.plot(feat_acc, 'go')
plt.errorbar(np.arange(4),feat_acc , [feat_acc - feat_acc_min, feat_acc_max - feat_acc], fmt='.k', ecolor='gray', lw=1)
plt.plot(np.arange(4)+0.2, test_acc, 'bo')
plt.errorbar(np.arange(4)+0.2, test_acc , [test_acc - test_acc_min, test_acc_max - test_acc], fmt='.k', ecolor='gray', lw=1)
plt.title('Test Accuracy')
plt.legend(['Handcrafted with ANN','Automatic with CNN']      ,loc=4 )
plt.xticks(np.arange(4), ['EEG','EEG+EOG','EEG+EMG','All'])
plt.figure(figsize=[6,3])
plt.plot(feat_f1, 'go')
plt.errorbar(np.arange(4),feat_f1 , [feat_f1 - feat_f1_min, feat_f1_max - feat_f1], fmt='.k', ecolor='gray', lw=1)
plt.plot(np.arange(4)+0.2, test_f1, 'bo')
plt.errorbar(np.arange(4)+0.2, test_f1 , [test_f1 - test_f1_min, test_f1_max - test_f1], fmt='.k', ecolor='gray', lw=1)
plt.title('Test F1')
plt.legend(['Handcrafted with ANN','Automatic with CNN']      ,loc=4 )
plt.xticks(np.arange(4), ['EEG','EEG+EOG','EEG+EMG','All'])

plt.figure(figsize=[5,3])
plt.plot(frec_acc, 'go')
plt.errorbar(np.arange(1),frec_acc , [frec_acc - frec_acc_min, frec_acc_max - frec_acc],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test Accuracy')
plt.xlim([-1,2])
plt.plot(0.2,rec_acc, 'bo')
plt.errorbar(np.arange(1)+0.2,rec_acc , [rec_acc - rec_acc_min, rec_acc_max - rec_acc],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test Accuracy Temporal')
plt.legend(['Handcrafted with RNN','Automatic with CNN+LSTM']      ,loc=4 )
plt.xticks(np.arange(1), ['All'])
plt.xlim([-1,2])
plt.figure(figsize=[5,3])
plt.plot(frec_f1, 'go')
plt.errorbar(np.arange(1),frec_f1 , [frec_f1 - frec_f1_min, frec_f1_max - frec_f1],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test Accuracy Temporal')
plt.xlim([-1,2])
plt.plot(0.2,rec_f1, 'bo')
plt.errorbar(np.arange(1)+0.2,rec_f1 , [rec_f1 - rec_f1_min, rec_f1_max - rec_f1],
             fmt='.k', ecolor='gray', lw=1)
plt.title('Test F1 Temporal')
plt.legend(['Handcrafted with RNN','Automatic with CNN+LSTM']      ,loc=4 )
plt.xticks(np.arange(1), ['All'])
plt.xlim([-1,2])