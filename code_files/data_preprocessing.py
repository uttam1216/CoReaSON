import mne
import numpy as np
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
from PIL import Image
from sklearn.model_selection import train_test_split
import sys
from utilities import *


# function to insert all data from TUH Seizure corpus to postgreSQL database in local for better EDA and data processing
def insert_data_in_postgres_tables(pat_ses_tcps):
    all_ctr = 0
    num_rows_inserted = 0
    num_rows_inserted_chnl_ann = 0
    num_rows_inserted_term_ann = 0
    try:
        for f in glob.glob(pat_ses_tcps):
            fol_id = f.split('/')[-1]
            pat_ses_tcps = f + '/*'
            for fin in glob.glob(pat_ses_tcps):
                pat_id = fin.split('/')[-1]
                ses_tcps = fin + '/*'
                for pat_fin in glob.glob(ses_tcps):
                    full_ses_id = pat_fin.split('/')[-1]
                    arr = full_ses_id.split('_')
                    ses_id = arr[0]
                    ses_date = arr[1] + '-' + arr[2] + '-' + arr[3]
                    ses_tcps = pat_fin + '/*'
                    cnt_tcps = 0
                    lst_dis_tcps = []
                    for pat_ses_fin in glob.glob(ses_tcps):
                        tcp_id = pat_ses_fin.split('/')[-1]
                        end_files = pat_ses_fin + '/*'
                        #we skip first 5 rows as they contain comments and metadata
                        for end_file in glob.glob(end_files):
                            t_id = end_file.split('/')[-1].split('.')[0].split('_')[-1]
                            file_type = ''
                            if end_file.split('.')[-1] == 'csv':
                                file_type = 'channel_annotation'
                                channel_df = pd.read_csv(end_file, skiprows=5)
                                for row in channel_df.itertuples(index=False):
                                    # insert data in channel_annotations table
                                    num_rows_inserted_chnl_ann += insert_in_channel_annotations(row.channel,
                                                                                                row.start_time,
                                                                                                row.stop_time,
                                                                                                row.label,
                                                                                                row.confidence, pat_id,
                                                                                                ses_id, t_id, tcp_id,
                                                                                                end_file, ses_date)
                            elif end_file.split('.')[-1] == 'csv_bi':
                                file_type = 'term_annotation'
                                term_df = pd.read_csv(end_file, skiprows=5)
                                for row in term_df.itertuples(index=False):
                                    # insert data in term_annotations table
                                    num_rows_inserted_term_ann += insert_in_term_annotations(row.channel,
                                                                                             row.start_time,
                                                                                             row.stop_time, row.label,
                                                                                             row.confidence, pat_id,
                                                                                             ses_id, t_id, tcp_id,
                                                                                             end_file, ses_date)
                            elif end_file.split('.')[-1] == 'edf':
                                file_type = 'edf'

                            # insert data in patient table
                            num_rows_inserted += insert_in_patient(fol_id, pat_id, ses_id, tcp_id, t_id, ses_date,
                                                                   end_file, file_type)
                            all_ctr += 1
    except Exception as ex:
        print(ex)


# Processing seiz & non seiz records for storage
def process_seiz_non_seiz_records_for_storage(fnsz_seiz_channel_df):
    # %%capture
    lst_main_excp, lst_pstr_label_excp, lst_exception, lst_pstr_label_elec_nsbst_nsast_done = [], [], [], []  # global variables
    ctr, num_rows_inserted_for_seiz, num_rows_inserted_for_non_seiz_bs, num_rows_inserted_for_non_seiz_as = 0, 0, 0, 0  # temp var just for testing and breaking after 1 record
    pstr, label = '', ''
    try:
        # for row in seiz_excpt_gnsz_channel_df.itertuples(index=False):   # EXCEPT GNSZ
        for row in fnsz_seiz_channel_df.itertuples(index=False):
            # unique patient session info
            pstr = row.patient_id + '$' + row.session_id + '_' + str(row.session_date).split('T')[0].replace('-',
                                                                                                             '_').replace(
                ' 00:00:00', '') + '$' + row.t_id + '$' + row.tcp_ref
            # non dependent electrode params just for insertion
            label, start_time, stop_time, confidence, file_path, dataset_type = get_non_depenedent_elect_params(
                row.label, row.start_time, row.stop_time, row.confidence, row.file_path)
            data = mne.io.read_raw_edf(file_path, preload=True)  # reading the file at hand in the df row
            raw = data.copy()
            lst_ch_names = raw.ch_names  # list of electrode names that this raw signal has in it
            # applying filters 0.5 to 40 Hz to get prominent epileptic features in the signal
            # raw = get_filtered_eeg(raw,freq_band=[0.5, 40]) ##commented on 12th Jul as Butterworth 0.1-64 is applied
            electrodes_arr = row.channel.split('-')
            if 1 == 1:  # ctr < 9:                 # for control during testing
                for electrode in electrodes_arr:
                    # fetch values for all electrode dependent attributes and then fire the insert statement
                    ref_electrode = fetch_ref_electrode(row.tcp_ref, electrode)
                    # in case ref_electrode is not present in the list of raw eeg electrodes then do not proceed further
                    if ref_electrode in lst_ch_names:
                        print('                                                                   ')
                        print('...STARTING TO PROCESS SEIZURE RECORD for electrode: ', ref_electrode)
                        # call function get_all_elec_dependent_params to get all params values
                        bos_vec, dos_vec, aos_vec, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, nn1, nn2, nn3, nn4, c_bos_nn1, c_dos_nn1, c_aos_nn1, c_bos_nn2, c_dos_nn2, c_aos_nn2, c_bos_nn3, c_dos_nn3, c_aos_nn3, c_bos_nn4, c_dos_nn4, c_aos_nn4, seiz_onset, non_seizure_occurrence, remarks = get_all_elec_dependent_params(
                            raw, ref_electrode, start_time, electrode, lst_ch_names, row.tcp_ref, 1, '', '')

                        ## insert values in table for each electrode
                        num_rows_inserted_for_seiz += insert_in_seiz_non_seiz(pstr, label, electrode, t1, t2, t3, t4,
                                                                              t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16,
                                                                              c_bos_nn1, c_dos_nn1, c_aos_nn1,
                                                                              c_bos_nn2, c_dos_nn2, c_aos_nn2,
                                                                              c_bos_nn3, c_dos_nn3, c_aos_nn3,
                                                                              c_bos_nn4, c_dos_nn4, c_aos_nn4, nn1, nn2,
                                                                              nn3, nn4, seiz_onset, start_time,
                                                                              stop_time, confidence, file_path,
                                                                              non_seizure_occurrence, dataset_type,
                                                                              remarks)

                        # print('......NOW PROCESS NON-SEIZURE RECORD FOR SAME ELECTRODE......')
                        # non_seiz_bs,non_seiz_as = create_non_seiz_rec_for_pstr_wd_sz_st_sp(seiz_excpt_gnsz_channel_df,pstr,label,confidence,start_time,stop_time,row.channel)  # EXCEPT GNSZ
                        non_seiz_bs, non_seiz_as = create_non_seiz_rec_for_pstr_wd_sz_st_sp(fnsz_seiz_channel_df, pstr,
                                                                                            label, confidence,
                                                                                            start_time, stop_time,
                                                                                            row.channel)
                        ### DO first for non_seiz_bs and then do for non_seiz_as for each electrode we give 2 records for non-seiz
                        ## DOING FOR non_seiz_bs
                        num_rows_inserted_for_non_seiz_bs += process_insertion_for_non_seiz(pstr, label, electrode,
                                                                                            non_seiz_bs, raw,
                                                                                            ref_electrode, lst_ch_names,
                                                                                            row.tcp_ref, 0, 'bs', '',
                                                                                            confidence, file_path,
                                                                                            dataset_type)

                        ## DOING FOR non_seiz_as
                        num_rows_inserted_for_non_seiz_as += process_insertion_for_non_seiz(pstr, label, electrode,
                                                                                            non_seiz_as, raw,
                                                                                            ref_electrode, lst_ch_names,
                                                                                            row.tcp_ref, 0, 'as', '',
                                                                                            confidence, file_path,
                                                                                            dataset_type)
            # ctr+=1
            # if ctr==2:
            #   break
    except Exception as ex:
        lst_main_excp.append(ex)
        lst_pstr_label_excp.append(pstr + label)


# process data insertion for non-seizure sessions records
def process_insertion_for_term_seiz_non_seizure_records(term_df, lst_valid_electrodes):
    # %%capture
    lst_main_excp, lst_exception, lst_bckg_excp, lst_pstrst_for_excp = [], [], [], []  # global variables
    ctr, num_rows_inserted_for_seiz, num_rows_inserted_for_bckg = 0, 0, 0  # temp var just for testing and breaking after 1 record
    try:
        for row in term_df.itertuples(index=False):
            # for each record, make entry of descretized values of all 21 electrodes
            pstr = row.pstr
            pstrst = pstr + '$' + str(row.start_time)  # NEW
            patient_id = pstr.split('$')[0]  # NEW
            # non dependent electrode params just for insertion
            label, start_time, stop_time, confidence, file_path, dataset_type = get_non_depenedent_elect_params(
                row.label, row.start_time, row.stop_time, row.confidence, row.file_path)
            term_seiz = 1 if label == 'seiz' else 0
            file_path = file_path.replace('edf_bi', 'edf')
            data = mne.io.read_raw_edf(file_path, preload=True)  # reading the file at hand in the df row
            raw = data.copy()
            lst_ch_names = raw.ch_names  # list of electrode names that this raw signal has in it
            ref_type = lst_ch_names[0].split('-')[-1]
            # raw = get_filtered_eeg(raw,freq_band=[0.5, 40]) ##commented on 12th Jul as Butterworth 0.1-64 is applied already in get_non_depenedent_elect_params
            for elec in lst_ch_names:
                e = elec.split('-')[0].split(' ')[-1]
                if '-REF' in elec and e in lst_valid_electrodes:
                    electrode = e
                    # fetch values for all electrode dependent attributes and then fire the insert statement
                    ref_electrode = fetch_ref_electrode(row.tcp_ref, electrode)
                    # in case ref_electrode is not present in the list of raw eeg electrodes then do not proceed further
                    if ref_electrode in lst_ch_names:
                        if term_seiz == 1:  ##print('...STARTING TO PROCESS TERM SEIZURE RECORDS...')
                            if start_time > 6 and stop_time > 12:
                                # call function get_all_elec_dependent_params to get all params values
                                bos_vec, dos_vec, aos_vec, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, nn1, nn2, nn3, nn4, c_bos_nn1, c_dos_nn1, c_aos_nn1, c_bos_nn2, c_dos_nn2, c_aos_nn2, c_bos_nn3, c_dos_nn3, c_aos_nn3, c_bos_nn4, c_dos_nn4, c_aos_nn4, seiz_onset, non_seizure_occurrence, remarks = get_all_elec_dependent_params(
                                    raw, ref_electrode, start_time, electrode, lst_ch_names, row.tcp_ref, 1, 'ts', ' ')
                                ## insert values in table for each electrode
                                try:
                                    num_rows_inserted_for_seiz += insert_in_term_seiz_non_seiz(pstrst, patient_id, pstr,
                                                                                               label, electrode, t1, t2,
                                                                                               t3, t4, t5, t6, t7, t8,
                                                                                               t9, t10, t11, t12, t13, t14, t15, t16,
                                                                                               c_bos_nn1, c_dos_nn1,
                                                                                               c_aos_nn1, c_bos_nn2,
                                                                                               c_dos_nn2, c_aos_nn2,
                                                                                               c_bos_nn3, c_dos_nn3,
                                                                                               c_aos_nn3, c_bos_nn4,
                                                                                               c_dos_nn4, c_aos_nn4,
                                                                                               nn1, nn2, nn3, nn4,
                                                                                               seiz_onset, start_time,
                                                                                               stop_time, confidence,
                                                                                               file_path,
                                                                                               non_seizure_occurrence,
                                                                                               dataset_type, remarks)
                                except Exception as ex:
                                    lst_exception.append(ex)
                                    pass
                        else:  ##print('......NOW PROCESS BCKG RECORDS ......')
                            lst_bckg_st_times = create_non_seiz_rec_for_pstrt_bckg(start_time, stop_time)
                            for st in lst_bckg_st_times:
                                if st > 6:  #### DO in loop for different start timings of bckg
                                    try:
                                        num_rows_inserted_for_bckg += process_insertion_for_bckg(pstrst, patient_id,
                                                                                                 pstr, label, electrode,
                                                                                                 st, raw, ref_electrode,
                                                                                                 lst_ch_names,
                                                                                                 row.tcp_ref, 0, 'bg',
                                                                                                 ' ', confidence,
                                                                                                 file_path,
                                                                                                 dataset_type)
                                    except Exception as ex:
                                        lst_bckg_excp.append(ex)
                                        pass
    except Exception as ex:
        lst_main_excp.append(ex)
        lst_pstrst_for_excp.append(pstrst)
        pass


def process_mean_non_seizures_correlations(descrtz_term_df, mean_non_seiz_corr_file):
    descrtz_bckg_term_df = descrtz_term_df.loc[descrtz_term_df['label'] == 'bckg']
    # pstrst is a unique key build up using patient session t_id reference type and start_time of seizure onset or non-seizure
    descrtz_bckg_term_df['pstrst'] = descrtz_bckg_term_df.apply(
        lambda row: (row['pstrst'].replace(str(row['pstrst'].split('$')[-1]), str(row['start_time']))), axis=1)
    # find patient wise mean of non seizures per electrode
    corr_rel_col = ['patient_id', 'electrode', 'nn1', 'nn2', 'nn3', 'nn4', 'c_bos_nn1', 'c_dos_nn1', 'c_aos_nn1',
                    'c_bos_nn2', 'c_dos_nn2', 'c_aos_nn2', 'c_bos_nn3', 'c_dos_nn3', 'c_aos_nn3', 'c_bos_nn4',
                    'c_dos_nn4', 'c_aos_nn4']
    corr_descrtz_bckg_term_df = descrtz_bckg_term_df[corr_rel_col]
    # CALL OF FUNC to get split nn dfs
    pat_elec_corr_descrtz_bckg_term_nn1_df, pat_elec_corr_descrtz_bckg_term_nn2_df, pat_elec_corr_descrtz_bckg_term_nn3_df, pat_elec_corr_descrtz_bckg_term_nn4_df = get_agg_split_nn_dfs(
        corr_descrtz_bckg_term_df)
    # Concatenate all 4 split dfs into one df
    nn_concat_df = pd.concat([pat_elec_corr_descrtz_bckg_term_nn1_df, pat_elec_corr_descrtz_bckg_term_nn2_df,
                              pat_elec_corr_descrtz_bckg_term_nn3_df, pat_elec_corr_descrtz_bckg_term_nn4_df])
    pat_elec_corr_descrtz_bckg_term_mean_df = nn_concat_df.groupby(['patient_id', 'electrode', 'nn']).agg(
        {'c_bos_nn': np.mean, 'c_dos_nn': np.mean, 'c_aos_nn': np.mean}).reset_index()
    pat_elec_corr_descrtz_bckg_term_mean_df['nn_nn'] = pat_elec_corr_descrtz_bckg_term_mean_df.apply(
        lambda row: (row['electrode'] + '_' + row['nn']), axis=1)
    # Because for bckg, bos, dos and aos are all same, we find mean of all as single mean correlation
    pat_elec_corr_descrtz_bckg_term_mean_df['mean_corr'] = pat_elec_corr_descrtz_bckg_term_mean_df.apply(
        lambda row: ((row['c_bos_nn'] + row['c_dos_nn'] + row['c_aos_nn']) / 3), axis=1)
    pat_elec_mean_corr_descrtz_bckg_term_df = pat_elec_corr_descrtz_bckg_term_mean_df[
        ['patient_id', 'electrode', 'nn', 'mean_corr']]
    # saving the non-seizures correlations mean for future use
    pat_elec_mean_corr_descrtz_bckg_term_df.to_csv(mean_non_seiz_corr_file,
                                                   index=False)
    return pat_elec_mean_corr_descrtz_bckg_term_df


# function for data clean up and also to add some statistical columns for further EDA
def process_data_cleanup_and_stats(fnsz_channel_df, term_df, output_folder_path):
    fnsz_channel_df = fnsz_channel_df.loc[fnsz_channel_df['start_time'] < fnsz_channel_df['stop_time']]
    # followig removes faulty patients records having seizure stop time earlier than seizure start time for same seizure e.g. issue for one such file has been reported by us to TUH EEG dataset provider by e-mail
    fnsz_channel_df = fnsz_channel_df.loc[fnsz_channel_df['start_time'] < fnsz_channel_df['stop_time']]
    fnsz_stats_df = get_seiz_stats_df(fnsz_channel_df)
    fnsz_stats_df['patient'] = fnsz_stats_df['pstr'].apply(lambda x: x.split('$')[0])
    fnsz_stats_df['ar_le'] = fnsz_stats_df.apply(lambda row: 'le' if 'le' in (row['pstr'].split('$')[-1]) else 'ar',
                                                 axis=1)
    # for each seiz_duration_top_chnl, find median and min and also at end for all
    fnsz_stats_df['dur_all_chnl_median'] = fnsz_stats_df['seiz_duration_all_chnl'].apply(lambda x: statistics.median(x))
    fnsz_stats_df['dur_all_chnl_min'] = fnsz_stats_df['seiz_duration_all_chnl'].apply(lambda x: min(x))
    # EDA done in another notebook, could not be shared due to presence of data on it and NDA with dataset provider
    # median duration of seizure over all session channels  46.6
    # min. duration of seizure over all session channels  1.85
    # mean duration of seizure over all session channels 58.809522703273494
    # remove seizures of duration less than equal to 3 seconds considering them as noise
    fnsz_stats_df = fnsz_stats_df.loc[fnsz_stats_df['dur_all_chnl_min'] > 3]
    # taking data with only Averaged reference montage and removing linked ear reference montage data
    fnsz_stats_df = fnsz_stats_df.loc[fnsz_stats_df['ar_le'] == 'ar']
    ## call function to frame a df with valid seizure onset times with-
    # contents: pstr, lst_seiz_start_times, unq_chnls_in_val_cnt having seiz_duration_all_chnl < 3 sec removed
    fnsz_onset_df = frame_seiz_onset_df(fnsz_stats_df, fnsz_channel_df, 'fnsz')
    # call function to formulate final seizure df with start and stop times
    final_fnsz_seiz_onset_df = formulate_final_seiz_onset_df(fnsz_onset_df, 'fnsz')
    # next we filter out this dataframe to have patients for whom normal control(non-seizure sessions) data is present
    final_fnsz_seiz_onset_df['patient_id'] = final_fnsz_seiz_onset_df.pstrst.apply(lambda x: x.split('$')[0])
    lst_unq_pat_with_fnsz_ar = list(final_fnsz_seiz_onset_df['patient_id'].unique())
    bckg_term_df = term_df.loc[(term_df['label'] == 'bckg') & (term_df['patient_id'].isin(lst_unq_pat_with_fnsz_ar))]
    bckg_term_df['ref_ar'] = bckg_term_df.tcp_ref.apply(
        lambda x: 1 if (x.split('_')[1] + '_' + x.split('_')[2]) == 'tcp_ar' else 0)
    bckg_term_ar_df = bckg_term_df.loc[(bckg_term_df['ref_ar'] == 1)]
    lst_pat_fnsz_bckg_ar = list(bckg_term_ar_df['patient_id'].unique())
    final_fnsz_seiz_onset_df['patient_id'] = final_fnsz_seiz_onset_df.pstrst.apply(lambda x: x.split('$')[0])
    final_fnsz_seiz_onset_df = final_fnsz_seiz_onset_df.loc[
        final_fnsz_seiz_onset_df.patient_id.isin(lst_pat_fnsz_bckg_ar)]
    # selecting only relevant columns for next step of train test split
    final_fnsz_seiz_onset_df = final_fnsz_seiz_onset_df[['pstrst', 'start_time', 'stop_time', 'file_path']]
    # we also make the non seizure df ready now
    bckg_term_ar_df['pstr'] = bckg_term_ar_df.apply(
        lambda row: formulate_pstr(row.patient_id, row.session_id, row.session_date, row.t_id, row.tcp_ref), axis=1)
    bckg_term_ar_df['dur'] = round((bckg_term_ar_df['stop_time'] - bckg_term_ar_df['start_time']), 2)
    # we remove non_seizure records less than 16 seconds in duration
    bckg_term_ar_df = bckg_term_ar_df.loc[
        bckg_term_ar_df['dur'] >= 16]  # 16 sec is our max time interval for extracting features
    # we sort the df in ascending order of duration and then start taking 16 sec records
    bckg_term_ar_df = bckg_term_ar_df.sort_values(by=['dur'])
    bckg_term_ar_df = bckg_term_ar_df[['pstr', 'start_time', 'stop_time', 'file_path', 'dur']]
    bckg_term_ar_df['posbl_rec'] = bckg_term_ar_df[
                                       'dur'] / 16  # 16 sec is our max time interval for extracting features but this does not mean that windows of > 16 sec are not possible
    bckg_term_ar_df['posbl_rec'] = bckg_term_ar_df['posbl_rec'].apply(lambda x: math.floor(x))
    bckg_term_ar_df['lst_s_e'] = bckg_term_ar_df.apply(
        lambda row: fetch_non_seiz_start_ends(row.start_time, row.posbl_rec), axis=1)
    # we call a function to prepare 10 times more non seizure time interval frames for our non seizure set than the fnsz seizure time intervals
    fnsz_final_non_seiz_df = formulate_final_non_seiz_onset_df(bckg_term_ar_df, 'fnsz')
    fnsz_final_non_seiz_df = fnsz_final_non_seiz_df.drop_duplicates()
    # revise the saved file
    fnsz_final_non_seiz_df.to_csv(output_folder_path + 'fnsz' + '_final_term_non_seiz_times.csv',
                                  index=False)
    fnsz_final_non_seiz_df = fnsz_final_non_seiz_df[['pstrst', 'start_time', 'stop_time', 'file_path']]
    lst_unq_non_seiz_pstrst = fnsz_final_non_seiz_df.pstrst.unique()
    # we intend to have only unique records in fnsz_final_non_seiz_df so we drop duplicates in next step
    fnsz_final_unq_non_seiz_df = pd.DataFrame()
    for item in lst_unq_non_seiz_pstrst:
        sel_df = fnsz_final_non_seiz_df.loc[fnsz_final_non_seiz_df['pstrst'] == item].iloc[0:1, :]
        # concat all such dfs to form one big df of size 20728
        fnsz_final_unq_non_seiz_df = pd.concat([fnsz_final_unq_non_seiz_df, sel_df])
    fnsz_final_unq_non_seiz_df['pstrst'] = fnsz_final_unq_non_seiz_df['pstrst'].apply(lambda x: x.replace('$', '__'))
    return final_fnsz_seiz_onset_df, fnsz_final_unq_non_seiz_df


if __name__ == '__main__':
    # Check if the correct number of arguments is provided
    try:
        if len(sys.argv) != 3:
            print("Usage: python3 data_preprocessing.py input_folder_path output_folder_path")
        else:
            input_folder_path = sys.argv[1]
            output_folder_path = sys.argv[2]
    except Exception as ex:
        input_folder_path = '/media/data/TUHEEG/tuh_eeg_seizure/v2.0.0/edf/*'  # some data source input path
        output_folder_path = '/media/data/ukumar/iBehave/data_files/'  # some output path
        pass
    # load all data from TUH EEG website into our local postgreSQL database
    insert_data_in_postgres_tables(
        input_folder_path)  # local path of the TUH EEG Seizure data folder as downloaded from the TUH website
    # load all patients data
    patient_df = load_df('patient')
    # load all channel annotations data
    channel_df = load_df('channel_annotations')
    # load all term annotations data
    term_df = load_df('term_annotations')
    # load all focal seizure channel annotations data into a dataframe
    fnsz_channel_df = channel_df.loc[channel_df['label'] == 'fnsz']
    # insert all seizure records in seiz_non_seiz table
    process_seiz_non_seiz_records_for_storage(fnsz_channel_df)
    # load EEG Graph data formulated manually in csv based on 10/20 standard electrode placement on human scalp for EEG recording
    electrodes_neighbor_df = fetch_neighbor_elec_mapping(
        output_folder_path + 'channels.csv')  # the file contains all 21 nodes along with its 1 hop connected neighbors..eg screenshot is in data_files
    lst_valid_electrodes = list(electrodes_neighbor_df['electrodes'])
    # load all seizure non-seizure term annotations data into the local database
    process_insertion_for_term_seiz_non_seizure_records(term_df, lst_valid_electrodes)
    # finding mean of non seizures for al patients to save it and use later in feature generation
    descrtz_term_df = load_df('term_seiz_non_seiz')
    pat_elec_mean_corr_descrtz_bckg_term_df = process_mean_non_seizures_correlations(descrtz_term_df,
                                                                                     output_folder_path + 'pat_elec_mean_corr_descrtz_bckg_term.csv')
    # call function to frame a final focal seizure onset dataframe
    final_fnsz_seiz_onset_df, fnsz_final_unq_non_seiz_df = process_data_cleanup_and_stats(fnsz_channel_df, term_df,
                                                                                          output_folder_path)
    # making train test split of the focal seizure onset dataset
    fnsz_onset_train_df, fnsz_onset_test_df = train_test_split(final_fnsz_seiz_onset_df, test_size=0.2)
    # saving the train and test files for feature extraction in anothe programs
    fnsz_onset_train_df.to_csv(output_folder_path + 'fnsz_onset_train.csv', index=False)
    fnsz_onset_test_df.to_csv(output_folder_path + 'fnsz_onset_test.csv', index=False)
    # making train test split of the non seizures dataset
    non_seiz_train_df, non_seiz_test_df = train_test_split(fnsz_final_unq_non_seiz_df, test_size=0.2)
    # saving the train and test files for feature extraction in anothe programs
    non_seiz_train_df.to_csv(output_folder_path + 'non_seiz_train.csv', index=False)
    non_seiz_test_df.to_csv(output_folder_path + 'non_seiz_test.csv', index=False)
    # from all these train test sets, we remove in total 420 records from 15 sessions for which the EEG images came out to be total black
    # this might be due to some technical issue, session ids and tids were noted but cannot be shared here due to NDA signed with TUHEEG dataset provider
