# -*- coding: utf-8 -*-

import os
from functions import loop_train_test
from data import image_size_dict as dims
from data import draw_false_color
import keras.backend as K
import tensorflow as tf

# remove abundant output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


## global constants
verbose = 1 # whether or not print redundant info (1 if and only if in debug mode, 0 in run mode)
run_times = 10 # random run times, recommend at least 10
output_map = True # whether or not output classification map
only_draw_label = True # whether or not only predict labeled samples
disjoint = False # whether or not train and test on spatially disjoint samples

lr = 1e-3 # init learing rate
decay = 1e-3 # exponential learning rate decay
ws = 19 # window size ori 19
epochs = 64 # epoch
batch_size = 32   # batch size
model_type = 'demo'  # model type

def pavia_university_experiment():
    hp = {
        'pc': dims['1'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    #num_list = [995,2797,315,460,202,754,200,552,142] #15%
    #num_list = [663,1865,210,306,135,503,133,368,95]  #10%
    #num_list = [332,933,105,153,68,252,67,184,48]     #5%
    num_list = [66,186,21,31,13,50,13,37,9]            #1%
    #num_list = [7, 17, 2, 3, 1, 5, 1, 4, 1]
    loop_train_test(dataID=1, num_list=num_list, verbose=verbose, run_times=run_times,
                    hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)

def indian_pine_experiment():
    hp = {
        'pc': dims['2'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    #num_list = [46,1428,830,237,483,730,28,478,20,974,2455,593,205,1265,386,93]  # 100% *16
    #num_list = [9,286,166,47,97,146,6,96,4,185,491,119,41,253,77,19]     #20%
    #num_list = [7,214,125,36,72,110,4,72,3,146,368,89,31,190,58,14]      #15%
    #num_list = [5, 143, 83, 24, 48, 73, 3, 48, 2, 97, 246, 59, 21, 127, 39, 9]  # 10%
    num_list = [2,71,42,12,24,37,1,24,1,49,123,30,10,63,19,5]      #5%
    #num_list = [1,14,8,2,5,7,1,5,1,10,25,6,2,13,4,1]  #1%
    

    loop_train_test(dataID=2, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)

def Salinas_experiment():
    hp = {
        'pc': dims['3'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    #num_list = [2009,3726,1976,1394,2678,3959,3579,11271,6203,3278,1068,1927,916,1070,7268,1807]  #100%
    #num_list = [100,]
    #num_list = [20, 37, 19, 13, 26, 39, 35, 112, 62, 32, 10, 19, 9, 10, 72, 18]   #1%
    num_list = [10, 19, 10, 7, 13, 20, 18, 56, 31, 16, 5, 10, 5, 5, 36, 9]#0.5%
    loop_train_test(dataID=4, num_list=num_list, verbose=verbose, run_times=run_times,
                    hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=False)


def WHU_Hi_HanChuan_experiment():
    hp = {
        'pc': dims['5'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }


    num_list =   [13, 13, 7, 12, 12, 3, 13, 12, 13, 12, 12, 12, 5, 4, 7, 8] #0.1%训练样本
    # num_list = [10, 19, 10, 7, 13, 20, 18, 56, 31, 16, 5, 10, 5, 5, 36, 9] #0.5%训练样本
    #num_list = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]  # 100训练样本
    # num_list = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200] #0.1%训练样本
    #num_list = [50] * 15  #每类固定使用50个训练样本
    loop_train_test(dataID=5, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)


def WHU_Hi_HongHu_experiment():
    hp = {
        'pc': dims['6'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }

    num_list = [140,35,218,1633,62,446,241,41,108,124,110,90,225,74,10,73,30,32,87,35,13,40]   #1%训练样本
    #num_list = [25] * 22  #每类固定使用25个训练样本
    num_list = [50] * 22  #每类固定使用50个训练样本
    loop_train_test(dataID=6, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)


def WHU_Hi_LongKow_experiment():
    hp = {
        'pc': dims['7'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }

    #num_list = [34511,8374,3031,63212,4151,11854,67056,7124,5229]#100%
    num_list = [28,6,2,51,3,10,54,6,4]
    #num_list = [35,8,3,63,4,12,67,7,5]#0.1%
    #num_list =  [13, 13, 7, 12, 12, 3, 13, 12, 13, 12, 12, 12, 5, 4, 7, 8] #0.1%训练样本
    #num_list = [10, 19, 10, 7, 13, 20, 18, 56, 31, 16, 5, 10, 5, 5, 36, 9] #0.5%训练样本
    #num_list = [20, 37, 19, 13, 26, 39, 35, 112, 62, 32, 10, 19, 9, 10, 72, 18]  # 1%训练样本
    # num_list = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200] #0.1%训练样本
    #num_list = [25] * 9  #每类固定使用25个训练样本
    #num_list = [50] * 9  #每类固定使用50个训练样本
    loop_train_test(dataID=7, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)


def houston_university_experiment():
    hp = {
        'pc': dims['3'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }

    num_list = [50] * 15  #每类固定使用50个训练样本
    loop_train_test(dataID=3, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)



#实验
#pavia_university_experiment()
#indian_pine_experiment()
#houston_university_experiment()
Salinas_experiment()
#WHU_Hi_HanChuan_experiment()
#WHU_Hi_HongHu_experiment()
#WHU_Hi_LongKow_experiment()

#draw_false_color(dataID=1)
#draw_false_color(dataID=2)
#draw_false_color(dataID=3)
draw_false_color(dataID=4)
#draw_false_color(dataID=5)
#draw_false_color(dataID=6)
#draw_false_color(dataID=7)

#
# draw_bar(dataID=1)
# draw_bar(dataID=2)
#draw_bar(dataID=3)

# draw_gt(dataID=1, fixed=disjoint)
# draw_gt(dataID=2, fixed=disjoint)
# draw_gt(dataID=3, fixed=disjoint)

