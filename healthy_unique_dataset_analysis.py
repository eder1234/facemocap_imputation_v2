from facemocap import FaceMoCap

mocap = FaceMoCap()
volunter_df = mocap.get_volunteer_mocap_data()
print(volunter_df)

if False:
    # now ignoring this
    sample_filename = volunter_df['filepath'][0]
    spc_sample = mocap.csv_to_spc(sample_filename)
    print(spc_sample.shape)
    valid_spc = mocap.remove_support_and_eyes(spc_sample)
    print(valid_spc.shape)
    print(mocap.count_complete_point_clouds(spc_sample))

    len_spc_list = []
    count_complete_pc = []
    for index, row in volunter_df.iterrows():
        filepath = row['filepath']
        spc = mocap.csv_to_spc(filepath)
        valid_spc = mocap.remove_support_and_eyes(spc)
        len_spc_list.append(valid_spc.shape[0])
        count_complete_pc.append(mocap.count_complete_point_clouds(valid_spc))

    volunter_df['len_spc'] = len_spc_list
    volunter_df['num_complete_pc'] = count_complete_pc

    volunter_df.to_csv('volunteer_data.csv',index=False)

total_pc = volunter_df['len_spc'].sum()
print(total_pc)
complete_pc = volunter_df['num_complete_pc'].sum()
print(complete_pc)
mean_pc = volunter_df['len_spc'].mean()
print(mean_pc)
std_pc = volunter_df['len_spc'].std()
print(std_pc)
max_len = volunter_df['len_spc'].max()
print(max_len)
min_len = volunter_df['len_spc'].min()
print(min_len)
