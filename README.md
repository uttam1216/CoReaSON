# Co-ReaSON
Correlation based Seizure Onset Detection <br>

The code in this repository is an implementation of our published approach: <br>
Uttam Kumar, Ran Yu, Michael Wenzel, and Elena Demidova. “Co-ReaSON: EEG-based Onset Detection of Focal Epileptic Seizures with Multimodal Feature Representations”. In: Proceedings of the 28th Pacific-Asia Conference on Knowledge Discovery and Data Mining, PAKDD 2024. LNAI. Springer, 2024, pp. 258–270. DOI: https://doi.org/10.1007/978-981-97-2238-9_20 <br> 
If you use the source code in your research, please cite our paper. <br>

<b>Notes: </b> <br>
1) This repository does not contain any data. We used TUH Seizure (TUSZ) dataset v2.0.0 for our research. To access this data, please directly contact the TUH Seizure (TUSZ) dataset author whose reference is: "Shah, V., et al.: The temple university hospital seizure detection corpus. Front. Neuroinform. 12, 83 (2018)" <br> <br>
2) Given the data, by following the below instructions, the codes provided can be run to exactly reach the results depicted in the submitted paper. <br>

<b> Repository Usage: </b> <br>
Installation: All Python packages can be installed by running the following command in your terminal- <br>
<b> <i> pip install -r requirements.txt </i> </b> <br>

<b> Data Preprocessing: </b> <br> Once you have the TUH EEG Seizure corpus at a location -<input_folder_path> and you are willing to keep all the processed data files at a location <output_folder_path>, then the data can be preprocessed by running the following command in your terminal- <br>
<b> <i> python3 data_preprocessing.py input_folder_path output_folder_path </i> </b> <br>

<b> Feature Extraction: </b> <br> Once you have the preprocessed data, the respective features can be extracted from it by running the following command in your terminal: <br>
<b> <i> python3 feature_extraction.py train_test_folder_path output_folder_path time_interval_window_length eeg_graph_nodes </i> </b> <br>
where train_test_folder_path is the path where you have the preprocessed data with train test files; output_folder_path is the path where you want to keep the files with extracted features; time_interval_window_length is an optional time interval window (4sec, 8sec, 12sec, 16sec or any time interval window (>=4 sec) of your choice for which you would like to run this model, default is 8 sec; and eeg_graph_nodes is the list of all nodes for which you would like to run our model, e.g. ['T3', 'T5', 'T4', 'T6'] or ['C3', 'CZ', 'C4'] or ['',''...''] which exists in our EEG Graph based on standard international 10-20 system for electrodes placement on human scalp for EEG recording. <br> 

<b> Predictive Model: </b> <br> Once you have the features extracted, the model can be run by running the following command in the terminal: <br>
<b> <i> python3 model.py train_dataset_path test_dataset_path num_epochs </i> </b> <br>
such that train_dataset_path is the path where training data is kept, test_dataset_path is the location of the test dataset, and num_epochs is an optional parameter giving the number of epochs for which you want to train the model. To change other parameters of the model, code can be changed in model.py for further research and experiments. <br>

<b> code_files </b> contains Python codes for data_preprocessing, feature extraction, model and an utilities file with many common functions.  <br>
<b> data_files </b> contains a Readme.txt file with instructions. <br>

<b>Disclaimer:</b>  This software is published only to support reproducibility in research. It is not intended for any type of commercial use or placement on the market.

<b>Affiliations: </b> The source code is being published in affiliation with the University of Bonn (https://www.uni-bonn.de/de). 

<b>Acknowledgements: </b> This work was partially funded by the Ministry of Culture and Science of the State of North Rhine-Westphalia, Germany (“iBehave”) and the Lamarr Institute for Machine Learning and Artificial Intelligence (https://lamarr-institute.org/).
