%%% Program for extrcating the label from EEG data (loaded using EEGLSB). 

clear all
clc
close all

%%%path, modelity, subject number and session
base_path='C:\Users\vibgup\Documents\EEG_StovckData\';
modelity={'EEG-rec', 'FMRI_rec'};
sub_num=3;
session=1;

%%if want to consider number of samples before\after the stimili
samples_after=50;

%%Number of Electrodes
electrodes=1:64;

%%%mapping numerical label into actual lable
ids = [111:114 125:128];
names = {'child','daughter','father','wife','four','three','ten','six'};
M = containers.Map(ids,names);

%%%Saving event.type information from loaded EEG data into variable category_info
%% event infromation contain the labels correspond to each trails (here 960 trails)
%% 111,112,113,114,125,126,127,128,1,2 are the lables where 1 and 2 correspond to start and relax time
category_info={EEG.event.type};


%%% event.latency contain the strating sample number for each trial.
%% For example, for trail 2 which labeled as 128, the latency is 7813 which is strating sample numbetr of trail 2
sample_info_wef_category=[EEG.event.latency];

%%%extract the unique labels
unique_cat=unique(category_info);

%%%store EEG data (loaded from EEGLAB) into variable sampls
sampls=EEG.data;

%%%mapping of stmuti on sample number
if length(category_info)>960
    start_ins=length(category_info)-960+1;
 else
   start_ins=1;
end

%%start and end of trails as for some cases there were more trails
end_ins=length(category_info);
sav_index=1;

sampls=sampls';sampls_per_entry=[];
for tt= start_ins:end_ins
    stmu_info=category_info(tt);
    stmu_info=cell2mat(stmu_info);
    stmu_info_com=stmu_info;
    if (stmu_info_com==1) | (stmu_info_com==2)
        disp('dont do anything')      
    else
        actual_label{sav_index}=M(stmu_info_com);
        samples_at_lable=sample_info_wef_category(tt)+samples_after;
        samples_at_next=sample_info_wef_category(tt+1)+samples_after;
            if (samples_at_next-samples_at_lable)==1024
                samples_at_next_up=samples_at_next-1;
            else
                samples_at_next_next=sample_info_wef_category(tt+2);
                len_nect_chunk=length(samples_at_lable:samples_at_next);
                additional_sample_needed=1024-len_nect_chunk;
                samples_at_next_up=samples_at_next+additional_sample_needed;
            end

        sampls_path=sampls(samples_at_lable:samples_at_next_up,electrodes);
        sampls_path=sampls_path;
        sampls_per_entry(sav_index,:)=sampls_path(:)';
        sav_index=sav_index+1;
    end

end


%%saving file and label
data.samples=sampls_per_entry;
data.label=categorical(actual_label);

file_save=[base_path,'processed_data\subject0',num2str(sub_num),'_session0',num2str(session),'.mat' ];
save(file_save, 'data');


