from utilities import *
import sys
import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, lfilter, sosfilt
import statistics
import math
import csv
import random
import matplotlib.pyplot as plt
import shutil
import os
import pywt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
# !pip install pyentrp
from pyentrp import entropy
from scipy.stats import kurtosis, skew, pearsonr
from PIL import Image
import pickle
import seaborn as sns
from matplotlib import rcParams


# Function to load all train test dataframes given a train test folder path as used in data_preprocessing.py
def load_train_test_dfs(train_test_folder_path):
    fnsz_onset_features_test_df = pd.read_csv(train_test_folder_path + 'fnsz_onset_test.csv')
    fnsz_onset_features_train_df = pd.read_csv(train_test_folder_path + 'fnsz_onset_train.csv')
    non_seiz_features_test_df = pd.read_csv(train_test_folder_path + 'non_seiz_test.csv')
    non_seiz_features_train_df = pd.read_csv(train_test_folder_path + 'non_seiz_train.csv')
    return fnsz_onset_features_test_df, fnsz_onset_features_train_df, non_seiz_features_test_df, non_seiz_features_train_df


# load EEG signal after passing it through notch filter to remove line noise from EEG
def load_notch_filtered_eeg(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.load_data()
    raw.notch_filter(freqs=50)
    return raw


# function to read edf file and extract/crop image of seizure-onset/non-seizure
def save_eeg_channel_images(edf_file_path, output_file_path, lst_channels_to_save, start_time, end_time):
    try:
        raw = load_notch_filtered_eeg(edf_file_path)
        raw.filter(l_freq=0.1, h_freq=64)
        raw.pick_channels(lst_channels_to_save)
        # Set the time period for which you want to save the images
        duration = end_time - start_time
        raw.crop(tmin=start_time, tmax=end_time)

        # Plot all EEG channels together
        plt = raw.plot(duration=duration)
        # plt.figure(figsize=(15, 8))
        plt.savefig(output_file_path)
        # plt.close()
    except Exception as ex:
        print('Exception came for file and start & end times: ', edf_file_path, start_time, end_time)
        print(ex)
        pass

# function to read edf file and extract/crop image of seizure-onset/non-seizure
def save_eeg_channel_images(edf_file_path, output_file_path, lst_channels_to_save, start_time, end_time):
        try:
            raw = load_notch_filtered_eeg(edf_file_path)
            raw.filter(l_freq=0.1, h_freq=64)
            raw.pick_channels(lst_channels_to_save)
            # Set the time period for which you want to save the images
            duration = end_time - start_time
            raw.crop(tmin=start_time, tmax=end_time)

            # Plot all EEG channels together
            plt = raw.plot(duration=duration)
            # plt.figure(figsize=(15, 8))
            plt.savefig(output_file_path)
            # plt.close()
        except Exception as ex:
            print('Exception came for file and start & end times: ', edf_file_path, start_time, end_time)
            print(ex)
            pass

# for a dataframe of format pstrst	start_time	stop_time	file_path, we save images of specific seizur eonset duration
def save_eeg_extracted_images(df,output_folder,lst_channels_to_save,seiz_non_seiz,window_dur):
    for row in df.itertuples(index=False):
        edf_file_path = row.file_path.replace('edf_bi','edf').replace('csv_bi','edf').replace('csv','edf')
        output_file_path = output_folder+row.pstrst+'.png'
        output_file_path = output_file_path.replace('$','__')
        if seiz_non_seiz == 'seiz':
           save_eeg_channel_images(edf_file_path, output_file_path, lst_channels_to_save, (row.start_time-(window_dur/2)), (row.start_time+(window_dur/2)))
        else:
           save_eeg_channel_images(edf_file_path, output_file_path, lst_channels_to_save, row.start_time, row.start_time+(window_dur)) #row.stop_time


### Apply Discrete Wavelet Transform and add Features for train test dfs
# func. to read edf & apply notch & butterworth filter & return digitalized signal inclusive of all chnls in list
def fetch_filtered_eeg_lst_chnls(raw, output_file_path, lst_channels_to_save, start_time, end_time):
    try:
        # print('After applying notch_filter: ',raw)
        raw.pick_channels(lst_channels_to_save)
        # print('After picking sel. chnls: ',raw)

        # Set the time period for which you want to save the images
        duration = end_time - start_time
        raw.crop(tmin=start_time, tmax=end_time)
        # print('After cropping: ',raw)

        # data_ch, times = raw.get_data(picks=ref_electrode, return_times=True, start=start_time, stop=end_time)
        # above one gives of single electrode, below one gives for a list of electrodes
        data_ch, times = raw.get_data(picks=lst_channels_to_save, return_times=True, start=start_time, stop=end_time)
        # print('data_ch.shape: ',data_ch.shape)

        #setting parameters for the application of butterworth filter
        lowcut, highcut, nyquist_freq, b_order = 0.1, 64, (raw.info['sfreq'] / 2.0), 2
        # print('nyquist_freq: ',nyquist_freq)
        # b_order is order of butterworth filter which we use as 2
        sos = butter(b_order, [lowcut / nyquist_freq, highcut / nyquist_freq], btype='band', output='sos')
        # print('sos: ',sos)
        ## Apply the filter to the signal
        filtered_signal = sosfilt(sos, data_ch)
        return filtered_signal
    except Exception as ex:
        print('Exception came for start & end times: ', start_time, end_time)
        print(ex)
        pass


# function to apply Discrete Wavelet Transform of a certain DB (e.g. 4 or 10) and level (e.g. 4) to the filtered eeg signal
def apply_wavelet_transform(signal, db, level):
    # Daubechies 10 wavelet transform with given level
    coeffs = pywt.wavedec(signal, db, level=level)
    # cA4, cD4, cD3, cD2, cD1 = coeffs
    return coeffs


# function to save and plot DB10A4 level 4 images for EEG from a given dataframe
def save_db10A4_eeg_images(df,output_file_path,lst_channels_to_save,window_dur):
    lst_excp_pstrst, ctr = [], 0
    for row in df.itertuples(index=False):
        try:
            file_name = row.pstrst.replace('$','__') + '.png'
            if not os.path.isfile(output_file_path+file_name):
               # i.e. only if file does not already exist, do following
               raw = load_notch_filtered_eeg(row.file_path.replace('edf_bi','edf').replace('csv_bi','edf').replace('csv','edf'))
               fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(raw, 'none', lst_channels_to_save, 0, math.floor(row.start_time+(window_dur/2)))
               #taking full window length of the signal
               fil_tf_inc_onset = fil_sig_for_elec[:,fil_sig_for_elec.shape[-1]-window_dur:fil_sig_for_elec.shape[-1]]
               # Apply wavelet transform DB10 Level 4
               wavelet_coeffs_db10 = apply_wavelet_transform(fil_tf_inc_onset, 'db10', 4)
               #A4 coefficients for whole window duration that needs to be mapped into image
               cA4 = wavelet_coeffs_db10[0]
               x = np.linspace(0, window_dur, 18)
               y = np.asarray(cA4)
               plt.plot(x, y.T)
               plt.xlabel("Time(s)")
               plt.ylabel("Elec (µV)")
               #saving the A4 mapped image to be used as one of the model inputs
               plt.savefig(output_file_path+file_name)
               plt.show()
               plt.close()
               ctr+=1
        except Exception as ex:
            print(ex)
            lst_excp_pstrst.append(row.pstrst)
            pass
    return ctr, lst_excp_pstrst


### Extracting features from DB10A4 applied on the filtered(notch+Butterworth) EEG Signal
# function to get Permutation Entropy, returns list of pd's
def get_permut_entropy(d1, d2, d3, d4, a4):
    pd1, pd2, pd3, pd4, pd5 = [], [], [], [], []
    [pd1.append(round(entropy.permutation_entropy(d1[i], 3, normalize=True), 2)) for i in range(0, d1.shape[0])]
    [pd2.append(round(entropy.permutation_entropy(d2[i], 3, normalize=True), 2)) for i in range(0, d2.shape[0])]
    [pd3.append(round(entropy.permutation_entropy(d3[i], 3, normalize=True), 2)) for i in range(0, d3.shape[0])]
    [pd4.append(round(entropy.permutation_entropy(d4[i], 3, normalize=True), 2)) for i in range(0, d4.shape[0])]
    [pd5.append(round(entropy.permutation_entropy(a4[i], 3, normalize=True), 2)) for i in range(0, a4.shape[0])]
    return pd1, pd2, pd3, pd4, pd5


# function to get Shannon Entropy, returns list of sh's
def get_shan_entropy(d1, d2, d3, d4, a4):
    sh1, sh2, sh3, sh4, sh5 = [], [], [], [], []
    [sh1.append(round(entropy.shannon_entropy(d1[i]), 2)) for i in range(0, d1.shape[0])]
    [sh2.append(round(entropy.shannon_entropy(d2[i]), 2)) for i in range(0, d2.shape[0])]
    [sh3.append(round(entropy.shannon_entropy(d3[i]), 2)) for i in range(0, d3.shape[0])]
    [sh4.append(round(entropy.shannon_entropy(d4[i]), 2)) for i in range(0, d4.shape[0])]
    [sh5.append(round(entropy.shannon_entropy(a4[i]), 2)) for i in range(0, a4.shape[0])]
    return sh1, sh2, sh3, sh4, sh5


# function to get Energy of EEG Signal, returns list of sh's
def get_energy(d1, d2, d3, d4, a4):
    en1, en2, en3, en4, en5 = [], [], [], [], []
    for i in range(0, d1.shape[0]):
        X = d1[i]
        summ = 0
        for i in X:
            summ += (i ** 2)
        en1.append(summ)
    for i in range(0, d2.shape[0]):
        X = d2[i]
        summ = 0
        for i in X:
            summ += (i ** 2)
        en2.append(summ)
    for i in range(0, d3.shape[0]):
        X = d3[i]
        summ = 0
        for i in X:
            summ += (i ** 2)
        en3.append(summ)
    for i in range(0, d4.shape[0]):
        X = d4[i]
        summ = 0
        for i in X:
            summ += (i ** 2)
        en4.append(summ)
    for i in range(0, a4.shape[0]):
        X = a4[i]
        summ = 0
        for i in X:
            summ += (i ** 2)
        en5.append(summ)

    return en1, en2, en3, en4, en5


# function to get statistical feature - mean of the DB10/DB4 with level4 of the EEG Signal
def get_meann(d1, d2, d3, d4, a4):
    mean1, mean2, mean3, mean4, mean5 = [], [], [], [], []
    [mean1.append(np.mean(d1[i])) for i in range(0, d1.shape[0])]
    [mean2.append(np.mean(d2[i])) for i in range(0, d2.shape[0])]
    [mean3.append(np.mean(d3[i])) for i in range(0, d3.shape[0])]
    [mean4.append(np.mean(d4[i])) for i in range(0, d4.shape[0])]
    [mean5.append(np.mean(a4[i])) for i in range(0, a4.shape[0])]
    return mean1, mean2, mean3, mean4, mean5


# function to get statistical feature - standard deviation of the DB10/DB4 with level4 of the EEG Signal
def get_stdd(d1, d2, d3, d4, a4):
    std1, std2, std3, std4, std5 = [], [], [], [], []
    [std1.append(np.std(d1[i])) for i in range(0, d1.shape[0])]
    [std2.append(np.std(d2[i])) for i in range(0, d2.shape[0])]
    [std3.append(np.std(d3[i])) for i in range(0, d3.shape[0])]
    [std4.append(np.std(d4[i])) for i in range(0, d4.shape[0])]
    [std5.append(np.std(a4[i])) for i in range(0, a4.shape[0])]
    return std1, std2, std3, std4, std5


# function to get statistical feature - skewness of the DB10/DB4 with level4 of the EEG Signal
def get_skeww(d1, d2, d3, d4, a4):
    skew1, skew2, skew3, skew4, skew5 = [], [], [], [], []
    [skew1.append(round(skew(d1[i]), 2)) for i in range(0, d1.shape[0])]
    [skew2.append(round(skew(d2[i]), 2)) for i in range(0, d2.shape[0])]
    [skew3.append(round(skew(d3[i]), 2)) for i in range(0, d3.shape[0])]
    [skew4.append(round(skew(d4[i]), 2)) for i in range(0, d4.shape[0])]
    [skew5.append(round(skew(a4[i]), 2)) for i in range(0, a4.shape[0])]
    return skew1, skew2, skew3, skew4, skew5


# function to get statistical feature - kurtosiss of the DB10/DB4 with level4 of the EEG Signal
def get_kurtosiss(d1, d2, d3, d4, a4):
    kurtosis1, kurtosis2, kurtosis3, kurtosis4, kurtosis5 = [], [], [], [], []
    [kurtosis1.append(round(kurtosis(d1[i]), 2)) for i in range(0, d1.shape[0])]
    [kurtosis2.append(round(kurtosis(d2[i]), 2)) for i in range(0, d2.shape[0])]
    [kurtosis3.append(round(kurtosis(d3[i]), 2)) for i in range(0, d3.shape[0])]
    [kurtosis4.append(round(kurtosis(d4[i]), 2)) for i in range(0, d4.shape[0])]
    [kurtosis5.append(round(kurtosis(a4[i]), 2)) for i in range(0, a4.shape[0])]
    return kurtosis1, kurtosis2, kurtosis3, kurtosis4, kurtosis5


# function to collect DWT related features of a filtered EEG signal data, returns list of 35 lists
def get_dwt_features(raw, out_fol, lst_channels_to_save, start_time, end_time, win_dur):
    fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(raw, out_fol, lst_channels_to_save, start_time, end_time)
    # for all channels, take last win_dur time interval number of seconds data points for applying debauchies
    fil_win_dur_inc_onset = fil_sig_for_elec[:, fil_sig_for_elec.shape[-1] - win_dur:fil_sig_for_elec.shape[-1]]
    # Apply wavelet transform DB10 Level 4 only for getting permuatation entropy
    wc_db10l4 = apply_wavelet_transform(fil_win_dur_inc_onset, 'db10', 4)
    pd1, pd2, pd3, pd4, pd5 = get_permut_entropy(wc_db10l4[4], wc_db10l4[3], wc_db10l4[2], wc_db10l4[1], wc_db10l4[0])
    # Apply wavelet transform DB4 Level 4 for other entropy and energy features
    wc_db4l4 = apply_wavelet_transform(fil_win_dur_inc_onset, 'db4', 4)
    #                                        D1,          D2,          D3,          D4,          A4
    sh1, sh2, sh3, sh4, sh5 = get_shan_entropy(wc_db4l4[4], wc_db4l4[3], wc_db4l4[2], wc_db4l4[1], wc_db4l4[0])
    sk1, sk2, sk3, sk4, sk5 = get_skeww(wc_db4l4[4], wc_db4l4[3], wc_db4l4[2], wc_db4l4[1], wc_db4l4[0])
    kt1, kt2, kt3, kt4, kt5 = get_kurtosiss(wc_db4l4[4], wc_db4l4[3], wc_db4l4[2], wc_db4l4[1], wc_db4l4[0])
    mn1, mn2, mn3, mn4, mn5 = get_meann(wc_db4l4[4], wc_db4l4[3], wc_db4l4[2], wc_db4l4[1], wc_db4l4[0])
    st1, st2, st3, st4, st5 = get_stdd(wc_db4l4[4], wc_db4l4[3], wc_db4l4[2], wc_db4l4[1], wc_db4l4[0])
    en1, en2, en3, en4, en5 = get_energy(wc_db4l4[4], wc_db4l4[3], wc_db4l4[2], wc_db4l4[1], wc_db4l4[0])
    X = [pd1, pd2, pd3, pd4, pd5, sh1, sh2, sh3, sh4, sh5, sk1, sk2, sk3, sk4, sk5, kt1, kt2, kt3, kt4, kt5,
         mn1, mn2, mn3, mn4, mn5, st1, st2, st3, st4, st5, en1, en2, en3, en4, en5]
    # each item of this list has 17 values in each item i.e. 1 value for each electrode
    return X


# function to get combination of features per electrode
def get_lst_chnlwise_features(lst_channels_to_save):
    # permutation entropy, shannon entropy, skewness, kurtosis, mean, standard deviation, energy
    lst_features = ['pd1', 'pd2', 'pd3', 'pd4', 'pd5', 'sh1', 'sh2', 'sh3', 'sh4', 'sh5', 'sk1', 'sk2', 'sk3', 'sk4',
                    'sk5',
                    'kt1', 'kt2', 'kt3', 'kt4', 'kt5', 'mn1', 'mn2', 'mn3', 'mn4', 'mn5', 'st1', 'st2', 'st3', 'st4',
                    'st5',
                    'en1', 'en2', 'en3', 'en4', 'en5']
    # constructing a list of new features to be made as columns of each of the dfs to store their values
    lst_chnlwise_feat = []  # lst_channels_to_save is total electrodes being considered, max=19
    for feature in lst_features:
        for elec_ref in lst_channels_to_save:
            elec = elec_ref.split(' ')[-1].split('-')[0]
            lst_chnlwise_feat.append(elec + '_' + feature)
    return lst_chnlwise_feat


# function to add features upon the train and test set dfs
def add_dwt_features_to_df(df, lst_channels_to_save, lst_chnlwise_feat, start_time_diff_factor, stop_time_diff_factor,
                           seiz_onset):
    # start_time_diff_factor - from what time before the start time should the signal be taken into consideration
    # its value is window time interval / 2
    # stop_time_diff_factor - until what time after the onset/non-seizure-start should the signal be considered
    # its value is window time interval / 2
    lst_excp, temp_pstrst = [], ''
    # since the images have __ instead of $ so we make a new col with __ instead of $
    df['uid'] = df['pstrst'].apply(lambda x: x.replace('$', '__'))
    # first we add all 595 columns as empty columns and later update values row-wise
    df = df.reindex(df.columns.tolist() + lst_chnlwise_feat, axis=1)
    for index, row in df.iterrows():
        temp_pstrst = row.pstrst
        # file_name = row.pstrst.replace('$','__') + '.png'
        raw = load_notch_filtered_eeg(row.file_path)
        try:
            F = get_dwt_features(raw, 'none', lst_channels_to_save, math.floor(row.start_time - start_time_diff_factor),
                                 math.floor(row.start_time + stop_time_diff_factor), start_time_diff_factor*2)
            lst_chnlwise_feat_val = []
            for lst_item in F:
                for ch_feat_val in lst_item:
                    lst_chnlwise_feat_val.append(ch_feat_val)
            for i in range(len(lst_chnlwise_feat)):
                df.at[index, lst_chnlwise_feat[i]] = lst_chnlwise_feat_val[i]
        except Exception as ex:
            lst_excp.append(temp_pstrst)
            pass
    df['label'] = seiz_onset
    return df, lst_excp


# function to get dwt features added in the dfs
def get_dwt_added_features_in_dfs(fnsz_onset_features_test_df, fnsz_onset_features_train_df, non_seiz_features_test_df,
                                  non_seiz_features_train_df, lst_channels_to_save, output_folder_path, win_dur):
    # list of features based on dwt column names per electrode for all selected electrodes
    lst_chnlwise_feat = get_lst_chnlwise_features(lst_channels_to_save)
    # get dfs with added dwt features, additionally get exception description if got any while making the features
    fnsz_onset_dwt_feat_test_df, lst_excp_seiz_test = add_dwt_features_to_df(fnsz_onset_features_test_df,
                                                                             lst_channels_to_save, lst_chnlwise_feat,
                                                                             win_dur / 2, win_dur / 2, 1)
    fnsz_onset_dwt_feat_train_df, lst_excp_seiz_train = add_dwt_features_to_df(fnsz_onset_features_train_df,
                                                                               lst_channels_to_save, lst_chnlwise_feat,
                                                                               win_dur / 2, win_dur / 2, 1)
    non_seiz_dwt_feat_test_df, lst_excp_non_seiz_test = add_dwt_features_to_df(non_seiz_features_test_df,
                                                                               lst_channels_to_save, lst_chnlwise_feat,
                                                                               win_dur / 2, win_dur / 2, 0)
    non_seiz_dwt_feat_train_df, lst_excp_non_seiz_train = add_dwt_features_to_df(non_seiz_features_train_df,
                                                                                 lst_channels_to_save,
                                                                                 lst_chnlwise_feat, win_dur / 2,
                                                                                 win_dur / 2, 0)
    # due to exceptions, some columns may get NaN values, so we need to remove such rows from the dfs
    valid_fnsz_onset_dwt_feat_test_df = fnsz_onset_dwt_feat_test_df.loc[
        ~fnsz_onset_dwt_feat_test_df['pstrst'].isin(lst_excp_seiz_test)]  # 1 record with NaN removed
    valid_fnsz_onset_dwt_feat_train_df = fnsz_onset_dwt_feat_train_df.loc[
        ~fnsz_onset_dwt_feat_train_df['pstrst'].isin(lst_excp_seiz_train)]  # 8 record with NaN removed
    valid_non_seiz_dwt_feat_test_df = non_seiz_dwt_feat_test_df.loc[
        ~non_seiz_dwt_feat_test_df['pstrst'].isin(lst_excp_non_seiz_test)]  # 81 record with NaN removed
    valid_non_seiz_dwt_feat_train_df = non_seiz_dwt_feat_train_df.loc[
        ~non_seiz_dwt_feat_train_df['pstrst'].isin(lst_excp_non_seiz_train)]  # 213 record with NaN removed
    # save the dwt features for each test train dfs of the seizure non-seizure
    valid_fnsz_onset_dwt_feat_test_df.to_pickle(output_folder_path + 'fnsz_onset_dwt_feat_test.pkl')
    valid_fnsz_onset_dwt_feat_train_df.to_pickle(output_folder_path + 'fnsz_onset_dwt_feat_train.pkl')
    valid_non_seiz_dwt_feat_test_df.to_pickle(output_folder_path + 'non_seiz_dwt_feat_test.pkl')
    valid_non_seiz_dwt_feat_train_df.to_pickle(output_folder_path + 'non_seiz_dwt_feat_train.pkl')
    return valid_fnsz_onset_dwt_feat_test_df, valid_fnsz_onset_dwt_feat_train_df, valid_non_seiz_dwt_feat_test_df, valid_non_seiz_dwt_feat_train_df


# function to print the correlations mean of seiz and bckg using seaborn on graph and save it for model input
def plot_corr_mean_comparison(df,out_file_path):
    rcParams['figure.figsize'] = 6, 4
    print('Curr Win Corr. vs Non-Seiz Mean Corr. Graph for : ', df.uid.values[0])
    df = df.reset_index()
    df = df[['nn_nn','curr_win_corr','non_seiz_mean_corr']]
    # convert to long (tidy) form
    dfm = df.melt('nn_nn', var_name='cols', value_name='correlation')
    #g = sns.catplot(x="nn_nn", y="vals", hue='cols', data=dfm, kind='point')
    sns.pointplot(x="nn_nn", y="correlation", hue='cols', data=dfm, height=10, aspect=0.8)
    #g._legend.remove()
    plt.savefig(out_file_path)
    plt.show()
    plt.close()
    print('---------------------------------------------------------------')
    print('                                                               ')


# function to save correlations features in case we remove nodes from a part of brain region
# the dataset being passed has already limited number of features as per the list of nodes for which it is desired by user
def save_corr_image(df,brain_reg_to_rem,out_data_folder,out_img_folder,ns_corr_mean_df,len_corr_features):
    # brain_reg_to_rem i svariable that indicated which brain region is to be removed ..i.e. P-Parietal,F-Frontal,T-Temporal,O-Occipital,C-Central,A- All to be retained
    # len_corr_features indicates total length of correlation features in dataframe columns ot be utilized for further processing
    tot_cnt, ctr = 0, 0
    # list of 1-hop nn electrodes to be taken from the dataframe
    lst_sig_nn_col = list(df.columns[-(len_corr_features+1):-1])
    # list of all possible i-hop nn electrodes - one brain region electrodes
    # Parietal P4 and Frontal e.g. FP1 to be distinguished correctly 'FP2-F8', 'F4-F8', 'P3-O1', 'P3-PZ'
    if brain_reg_to_rem == 'P':
       new_lst = []
       for item in lst_sig_nn_col:
           if 'FP' in item:
              new_lst.append(item)  # Because when FP is there P cannot be there as its neighbor
           elif 'P' in item:
              None
           else:
              new_lst.append(item)
       lst_sig_nn_col = new_lst
    else:
       lst_sig_nn_col = [x for x in lst_sig_nn_col if brain_reg_to_rem not in x]
    lst_req_cols = ['uid'] + lst_sig_nn_col
    sel_df = df[lst_req_cols]
    for row in sel_df.itertuples(index=False):
        tot_cnt+=1
        lst_sig_nn_col_vals, lst_sig_nn_col_ns_mean_vals = [], []
        # for this same patient find mean values for same columns (elec pairs)
        patient_id = row.uid.split('__')[0]
        sel_pat_non_seiz_corr_mean_df = ns_corr_mean_df.loc[ns_corr_mean_df['patient_id']==patient_id]
        if len(sel_pat_non_seiz_corr_mean_df) > 0:
           for i in range(len(lst_sig_nn_col)):
               lst_sig_nn_col_vals.append(row[-(len(lst_sig_nn_col))+i])
               mean_ns_val = None
               mean_ns_val = round(sel_pat_non_seiz_corr_mean_df.loc[sel_pat_non_seiz_corr_mean_df['nn_nn']==lst_sig_nn_col[i]]['mean_corr'].values[0],2)
               if mean_ns_val is None:
                  mean_ns_val = round(sel_pat_non_seiz_corr_mean_df.loc[sel_pat_non_seiz_corr_mean_df['nn_nn_rev']==lst_sig_nn_col[i]]['mean_corr'].values[0],2)
               lst_sig_nn_col_ns_mean_vals.append(mean_ns_val)
           # form a dataframe of these lists to then form an image
           temp_pdf_data = {'nn_nn': lst_sig_nn_col, 'curr_win_corr': lst_sig_nn_col_vals, 'non_seiz_mean_corr': lst_sig_nn_col_ns_mean_vals}
           pat_wise_corr_df = pd.DataFrame(temp_pdf_data, columns = ['nn_nn','curr_win_corr','non_seiz_mean_corr'])
           pat_wise_corr_df['uid'] = row.uid
           pat_wise_corr_df.to_csv(out_data_folder+str(row.uid)+'.pkl', index=False)
           plot_corr_mean_comparison(pat_wise_corr_df,out_img_folder+str(row.uid)+'.png')
           ctr+=1
    return tot_cnt, ctr


# following function adds correlation features and then saves images for given EEG channels as per df
def add_sig_corr_feat(df, lst_channels_to_save, tem_cen_nn, win_dur, seiz_onset):
    lst_excp_pstrst, ctr = [], 0
    df.drop('label', axis=1, inplace=True)
    df = df.reindex(df.columns.tolist() + tem_cen_nn, axis=1)
    for index, row in df.iterrows():
        tem_cen_nn_vals = []
        try:
            if 1 == 1:  # not os.path.isfile(output_file_path):
                # i.e. only if file does not already exist, do following
                raw = load_notch_filtered_eeg(row.file_path)
                fil_sig_for_elec = fetch_filtered_eeg_lst_chnls(raw, 'none', lst_channels_to_save, 0,
                                                                math.floor(row.start_time + win_dur/2))
                ##values for signal to be taken for teh entire window length
                fil_win_dur_inc_onset = fil_sig_for_elec[:, fil_sig_for_elec.shape[-1] - win_dur:fil_sig_for_elec.shape[-1]]
                for elec_comb in tem_cen_nn:
                    elec_1 = elec_comb.split('-')[0]
                    elec_2 = elec_comb.split('-')[-1]
                    idx_e1 = lst_channels_to_save.index('EEG ' + elec_1 + '-REF')
                    idx_e2 = lst_channels_to_save.index('EEG ' + elec_2 + '-REF')
                    #finding pearson correlation coefficient for the given window length time interval
                    temp_corr, p_value = pearsonr(fil_win_dur_inc_onset[idx_e1].ravel(),
                                                  fil_win_dur_inc_onset[idx_e2].ravel())
                    tem_cen_nn_vals.append(round(temp_corr, 2))
                for i in range(len(tem_cen_nn)):
                    df.at[index, tem_cen_nn[i]] = tem_cen_nn_vals[i]
                df['label'] = seiz_onset
        except Exception as ex:
            print(ex)
            ctr += 1
            lst_excp_pstrst.append(row.pstrst)
            pass
    return df, ctr, lst_excp_pstrst


# function to have brain region wise possible one hop neighbors as per the EEG Graph - 10/20 system of electrodes placement on human scalp
def get_lst_participating_brain_part_nn(brain_part_lst, st_all_nn):
    lst_brain_part_participat_nn = []
    for elec in brain_part_lst:
        for item in st_all_nn:
            if elec in item:
                lst_brain_part_participat_nn.append(item)
    # if no 1 hop neighbor come, we return all possible 1 hop neighbors as deafult
    if len(lst_brain_part_participat_nn) == 0:
        lst_brain_part_participat_nn = st_all_nn
    return lst_brain_part_participat_nn


# function to get dwt features added in the dfs and save the built pickle file
def get_corr_added_features_in_dfs(valid_fnsz_onset_dwt_feat_test_df, valid_fnsz_onset_dwt_feat_train_df,
                                   valid_non_seiz_dwt_feat_test_df, valid_non_seiz_dwt_feat_train_df,
                                   output_folder_path, lst_channels_to_save, one_hop_nn, win_dur):
    valid_fnsz_onset_corr_feat_test_df, c, lst_excp = add_sig_corr_feat(valid_fnsz_onset_dwt_feat_test_df,
                                                                        lst_channels_to_save, one_hop_nn, win_dur, 1)
    # c=30, removing all rows where exception came due to NaN encountered while finding correlations with node vectors of certain time intervals
    valid_fnsz_onset_corr_feat_test_df = valid_fnsz_onset_corr_feat_test_df.loc[
        ~valid_fnsz_onset_corr_feat_test_df['pstrst'].isin(lst_excp)]
    # save the df into .pkl file for future use
    valid_fnsz_onset_corr_feat_test_df.to_pickle(output_folder_path + 'corr/fnsz_onset_corr_feat_test.pkl')

    valid_fnsz_onset_corr_feat_train_df, c_ts, lst_excp_c_ts = add_sig_corr_feat(valid_fnsz_onset_dwt_feat_train_df,
                                                                                 lst_channels_to_save, one_hop_nn, win_dur, 1)
    # c_ts = 108, removing all rows where exception came due to NaN encountered while finding correlations with node vectors of certain time intervals
    valid_fnsz_onset_corr_feat_train_df = valid_fnsz_onset_corr_feat_train_df.loc[
        ~valid_fnsz_onset_corr_feat_train_df['pstrst'].isin(lst_excp_c_ts)]
    # save the df into .pkl file for future use
    valid_fnsz_onset_corr_feat_train_df.to_pickle(output_folder_path + 'corr/fnsz_onset_corr_feat_train.pkl')

    valid_non_seiz_corr_feat_test_df, c_tns, lst_excp_tns = add_sig_corr_feat(valid_non_seiz_dwt_feat_test_df,
                                                                              lst_channels_to_save, one_hop_nn, win_dur, 0)
    # c_tns = 12, removing all rows where exception came due to NaN encountered while finding correlations with node vectors of certain time intervals
    valid_non_seiz_corr_feat_test_df = valid_non_seiz_corr_feat_test_df.loc[
        ~valid_non_seiz_corr_feat_test_df['pstrst'].isin(lst_excp_tns)]
    # save the df into .pkl file for future use
    valid_non_seiz_corr_feat_test_df.to_pickle(output_folder_path + 'corr/fnsz_ns_corr_feat_test.pkl')

    valid_non_seiz_corr_feat_train_df, c_trns, lst_excp_trns = add_sig_corr_feat(valid_non_seiz_dwt_feat_train_df,
                                                                                 lst_channels_to_save, one_hop_nn, win_dur, 0)
    # c_trns=47, removing all rows where exception came due to NaN encountered while finding correlations with node vectors of certain time intervals
    valid_non_seiz_corr_feat_train_df = valid_non_seiz_corr_feat_train_df.loc[
        ~valid_non_seiz_corr_feat_train_df['pstrst'].isin(lst_excp_trns)]
    # save the df into .pkl file for future use
    valid_non_seiz_corr_feat_train_df.to_pickle(output_folder_path + 'corr/fnsz_ns_corr_feat_train.pkl')

    return valid_fnsz_onset_corr_feat_test_df, valid_fnsz_onset_corr_feat_train_df, valid_non_seiz_corr_feat_test_df, valid_non_seiz_corr_feat_train_df

#function to save eeg images for given number of nodes for all dfs(i.e. test train seiz non-seiz)
def save_eeg_images_for_dfs(valid_fnsz_onset_corr_feat_test_df, valid_fnsz_onset_corr_feat_train_df, valid_non_seiz_corr_feat_test_df, valid_non_seiz_corr_feat_train_df, output_folder_path, lst_channels_to_save, win_dur):
    save_eeg_extracted_images(valid_fnsz_onset_corr_feat_test_df, output_folder_path+'eeg/test/sz/', lst_channels_to_save, 'seiz', win_dur)
    save_eeg_extracted_images(valid_fnsz_onset_corr_feat_train_df, output_folder_path + 'eeg/train/sz/', lst_channels_to_save, 'seiz', win_dur)
    save_eeg_extracted_images(valid_non_seiz_corr_feat_test_df, output_folder_path + 'eeg/test/ns/', lst_channels_to_save, 'non_seiz', win_dur)
    save_eeg_extracted_images(valid_non_seiz_corr_feat_train_df, output_folder_path + 'eeg/train/ns/', lst_channels_to_save, 'non_seiz', win_dur)


# function to save correlation images of different test train seizure non-seizure dfs for given number of channels(LENGTH OF CORR COLUMNS TO BE PLOTTED TO BE CHECKED) and window duration
def save_correlation_graph_images_for_dfs(valid_fnsz_onset_corr_feat_test_df, valid_fnsz_onset_corr_feat_train_df, valid_non_seiz_corr_feat_test_df, valid_non_seiz_corr_feat_train_df, out_data_folder, len_corr_features):
        # next to save the correlation features into images for model input, we first load non seizure mean df also as computed during data processing as a mean of non-seizure duration per patient
        pat_elec_mean_corr_descrtz_bckg_term_df = pd.read_csv(
            out_data_folder+'other_files/pat_elec_mean_corr_descrtz_bckg_term.csv')
        pat_elec_mean_corr_descrtz_bckg_term_df['nn_nn'] = pat_elec_mean_corr_descrtz_bckg_term_df.apply(
            lambda row: (row['electrode'] + '-' + row['nn']), axis=1)
        pat_elec_mean_corr_descrtz_bckg_term_df['nn_nn_rev'] = pat_elec_mean_corr_descrtz_bckg_term_df.apply(
            lambda row: (row['nn'] + '-' + row['electrode']), axis=1)
        tot_cnt_stest, ctr_stest = save_corr_image(valid_fnsz_onset_corr_feat_test_df, 'A', out_data_folder+'corr/foc_test/', out_data_folder+'corr/test/sz/', pat_elec_mean_corr_descrtz_bckg_term_df, len_corr_features)
        tot_cnt_strain, ctr_strain = save_corr_image(valid_fnsz_onset_corr_feat_train_df,'A', out_data_folder+'corr/foc_train/',out_data_folder+'corr/train/sz/', pat_elec_mean_corr_descrtz_bckg_term_df, len_corr_features)
        tot_cnt_nstest, ctr_nstest = save_corr_image(valid_non_seiz_corr_feat_test_df, 'A', out_data_folder+'corr/fnsc_test/', out_data_folder+'corr/test/ns/', pat_elec_mean_corr_descrtz_bckg_term_df, len_corr_features)
        tot_cnt_nstrain, ctr_nstrain = save_corr_image(valid_non_seiz_corr_feat_train_df, 'A', out_data_folder+'corr/fnsc_train/', out_data_folder+'corr/train/ns/', pat_elec_mean_corr_descrtz_bckg_term_df, len_corr_features)
        # counters returned by save_corr_image could be printed to get number of images saved

def save_dwt_images_for_dfs(valid_fnsz_onset_corr_feat_test_df, valid_fnsz_onset_corr_feat_train_df,
                            valid_non_seiz_corr_feat_test_df, valid_non_seiz_corr_feat_train_df, output_folder_path, lst_channels_to_save,
                            win_dur):
    num_suc_img_sz_test, lst_excp_sz_test = save_db10A4_eeg_images(valid_fnsz_onset_corr_feat_test_df, output_folder_path + 'dwt/test/sz/', lst_channels_to_save, win_dur)
    num_suc_img_sz_train, lst_excp_sz_train = save_db10A4_eeg_images(valid_fnsz_onset_corr_feat_train_df, output_folder_path + 'dwt/train/sz/', lst_channels_to_save, win_dur)
    num_suc_img_ns_test, lst_excp_ns_test = save_db10A4_eeg_images(valid_non_seiz_corr_feat_test_df, output_folder_path + 'dwt/test/ns/', lst_channels_to_save, win_dur)
    num_suc_img_ns_train, lst_excp_ns_train = save_db10A4_eeg_images(valid_non_seiz_corr_feat_train_df, output_folder_path + 'dwt/train/ns/', lst_channels_to_save, win_dur)
    # number of images could be printed for each df using the returned counters of save_db10A4_eeg_images function

# function to combine all three types of images for model input
def combine_all_images(image_path1, image_path2, image_path3, output_path):
    #opening the images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)
    image3 = Image.open(image_path3)
    #getting the dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size
    width3, height3 = image3.size
    #determining the size of combined image
    combined_width = width1 + width2  # + width3
    combined_height = max(height1, height2, height3)
    #creating a new image with the calculated size
    combined_image = Image.new('RGBA', (combined_width, combined_height), (0, 0, 0, 0))
    #pasting the first image onto the combined image
    combined_image.paste(image1, (0, 0))
    #pasting the second image onto the combined image
    combined_image.paste(image2, (width1, 0))
    #pasting the third image onto the combined image
    combined_image.paste(image3, (width1, height2))
    #saving the combined image
    combined_image.save(output_path)
    # im = Image.open(output_path)
    # im.show()

# function to combine three types of input images into one and save for model input
def combine_brain_parts_images(corr_lst):
    for i in range(len(corr_lst)):
        for image_path3 in glob.glob(corr_lst[i]):
            image_path2 = image_path3.replace('corr','dwt')
            image_path1 = image_path3.replace('corr','eeg')
            output_path = image_path3.replace('corr','all_cb')
            # if all three types of input images are present in their respective folders then only combine them and save
            if os.path.isfile(image_path3) and os.path.isfile(image_path2) and os.path.isfile(image_path1):
               combine_all_images(image_path1, image_path2, image_path3, output_path)

# function to create lists of path for combining and saving the tree types of images
def create_lists_of_image_paths():
    db10_lst = [output_folder_path+'dwt/test/sz/*.png', output_folder_path+'dwt/test/ns/*.png', output_folder_path+'dwt/train/sz/*.png', output_folder_path+'dwt/train/ns/*.png']
    corr_lst = [output_folder_path+'corr/test/sz/*.png', output_folder_path+'corr/test/ns/*.png', output_folder_path+'corr/train/sz/*.png', output_folder_path+'corr/train/ns/*.png']
    raw_lst = [output_folder_path+'eeg/test/sz/*.png', output_folder_path+'eeg/test/ns/*.png', output_folder_path+'eeg/train/sz/*.png', output_folder_path+'eeg/train/ns/*.png']
    cb_lst = [output_folder_path+'all_cb/test/sz/', output_folder_path+'all_cb/test/ns/', output_folder_path+'all_cb/train/sz/', output_folder_path+'all_cb/train/ns/']
    return db10_lst, corr_lst, raw_lst, cb_lst


if __name__ == '__main__':
    # list of EEG Graph nodes presented in lists as per different brain regions
    lst_temporal, lst_central, lst_frontal, lst_parietal, lst_occipital = ['T3', 'T5', 'T4', 'T6'], ['C3', 'CZ', 'C4'], ['F3', 'F7', 'FP1', 'FZ', 'FP2', 'F4', 'F8'], ['P3', 'PZ', 'P4'], ['O1', 'O2']
    lst_all = lst_temporal + lst_central + lst_frontal + lst_parietal + lst_occipital
    # all possible one hop nearest neighbor node combinations
    all_nn = ['FP1-FP2', 'F7-FP1', 'F3-FP1', 'F4-FP2', 'F8-FP2', 'F7-F3', 'F3-FZ', 'FZ-F4', 'F4-F8',
              'T3-F7', 'C3-F3', 'CZ-FZ', 'C4-F4', 'T4-F8', 'A1-T3', 'T3-C3', 'C3-CZ',
              'CZ-C4', 'C4-T4', 'T4-A2', 'T3-T5', 'C3-P3', 'CZ-PZ', 'C4-P4', 'T4-T6', 'T5-P3', 'P3-PZ',
              'PZ-P4', 'P4-T6', 'T5-O1', 'P3-O1', 'P4-O2', 'T6-O2', 'O1-O2']  #34 1-hop nn i.e. Edges as per our EEG Graph
    # if we remove A1, A2 nodes with their neighbor
    #all_nn = ['C3-CZ', 'C3-P3', 'C3-T3', 'C3-F3', 'C4-P4', 'C4-T4', 'CZ-C4', 'C4-F4', 'CZ-FZ', 'CZ-PZ',
    #          'T3-T5', 'T4-T6', 'P3-T5', 'P4-T6', 'T3-F7', 'T4-F8', 'T5-O1', 'T6-O2',
    #          'F3-FP1', 'F3-F7', 'F3-FZ', 'F7-FP1', 'FP1-FP2', 'FZ-F4', 'FP2-F4', 'FP2-F8', 'F4-F8',
    #          'P3-O1', 'P3-PZ', 'PZ-P4', 'P4-O2', 'O1-O2']

    # Check if correct number of arguments is provided
    if len(sys.argv) < 3:
        print('Please at least give train_test_folder_path and output_folder_path to proceed!')
        print(
            "Usage: python3 feature_extraction.py train_test_folder_path output_folder_path time_interval_window_length eeg_graph_nodes")
        sys.exit()
    else:
        try:
            train_test_folder_path = sys.argv[1]
            output_folder_path = sys.argv[2]
            win_dur = sys.argv[3]  # time_interval_window_length can be 4,8,12 or 16 seconds
            lst_nodes = sys.argv[
                4]  # brain_regional_nodes should always be a list of nodes to be considered for seizure onset detection
            # e.g. lst_all, all_temporal, all_frontal, all_central, all_parietal, all_occipital
        except Exception as ex:
            if len(sys.argv) == 3:
                win_dur = 8  # by default we keep it as 8 for CoReaSON's best performance, even though it works for any given input window
                lst_nodes = lst_all
            else:
                lst_nodes = lst_all
            pass
    # Loading all train test data in seizure non-seizure respective data frames
    fnsz_onset_features_test_df, fnsz_onset_features_train_df, non_seiz_features_test_df, non_seiz_features_train_df = load_train_test_dfs(
        train_test_folder_path)
    # list of all electrodes # max - 19, min - as per a brain region in EEG Graph

    # fetch_ref_electrode is a function in utilities that returns electrode in correct format as per given tcp montage
    lst_channels_to_save = [fetch_ref_electrode('tcp_ar', electrode) for electrode in lst_nodes]

    # call function to perform DWT of the EEG signal and then take DWT statistical features added in the train test dfs
    valid_fnsz_onset_dwt_feat_test_df, valid_fnsz_onset_dwt_feat_train_df, valid_non_seiz_dwt_feat_test_df, valid_non_seiz_dwt_feat_train_df = get_dwt_added_features_in_dfs(
        fnsz_onset_features_test_df, fnsz_onset_features_train_df, non_seiz_features_test_df,
        non_seiz_features_train_df, lst_channels_to_save, output_folder_path, win_dur)

    # get list of brain region wise one hop nearest neighbors...lst_nodes can have electrodes in number from 1 to max 19 as per our EEG Graph defined in paper
    lst_brain_parts_nn = get_lst_participating_brain_part_nn(lst_nodes, all_nn)
    # we find the number of 1 hop nearest neighbors for which we will compute the correlations so as to use them in future
    len_corr_columns = len(lst_brain_parts_nn)

    # call function to get all correlations features added in the train test dfs for the given channels and given window
    valid_fnsz_onset_corr_feat_test_df, valid_fnsz_onset_corr_feat_train_df, valid_non_seiz_corr_feat_test_df, valid_non_seiz_corr_feat_train_df = get_corr_added_features_in_dfs(
        valid_fnsz_onset_dwt_feat_test_df, valid_fnsz_onset_dwt_feat_train_df, valid_non_seiz_dwt_feat_test_df,
        valid_non_seiz_dwt_feat_train_df, output_folder_path, lst_channels_to_save, lst_brain_parts_nn, win_dur)

    # returned dataframes now are rich in both i.e. dwt features + correlation features
    # next we save the dataframe wise correlations images into graphs which will be used as model input
    save_correlation_graph_images_for_dfs(valid_fnsz_onset_corr_feat_test_df, valid_fnsz_onset_corr_feat_train_df, valid_non_seiz_corr_feat_test_df, valid_non_seiz_corr_feat_train_df, output_folder_path, len_corr_columns)

    # save eeg images for given number of nodes for a window for all dfs(i.e. test train seiz non-seiz)
    save_eeg_images_for_dfs(valid_fnsz_onset_corr_feat_test_df, valid_fnsz_onset_corr_feat_train_df, valid_non_seiz_corr_feat_test_df, valid_non_seiz_corr_feat_train_df, output_folder_path, lst_channels_to_save, win_dur)

    # save DWT A4 coefficient images for given number of nodes for a window for all dfs(i.e. test train seiz non-seiz)
    save_dwt_images_for_dfs(valid_fnsz_onset_corr_feat_test_df, valid_fnsz_onset_corr_feat_train_df, valid_non_seiz_corr_feat_test_df, valid_non_seiz_corr_feat_train_df, output_folder_path, lst_channels_to_save, win_dur)

    # Combine all three types of generated images for Model Input
    # list of paths for loading three types of input images for combining and then keeping the combined images to a destined path
    db10_lst, corr_lst, raw_lst, cb_lst = create_lists_of_image_paths(output_folder_path)

    #call function to process the combination of the input images and save them
    combine_brain_parts_images(corr_lst) #we pass only corr_lst as by replaceing a folder name we use files of dwt, raw and combined ones
