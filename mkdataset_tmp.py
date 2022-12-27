#%%
import os
import shutil

#%%
# move the 800-900 images from train to test_tmp:
# "DIV2K_test_tmp_HR", "DIV2K_test_tmp_LR_bicubic"

root_path = '/data/Cici1996/DIV2K'
# source_path_HR = os.path.join(root_path, 'DIV2K_train_HR')
source_path_LR = os.path.join(root_path, 'DIV2K_train_LR_bicubic/X4')
# HR_name_list = os.listdir(source_path_HR)
LR_name_list = os.listdir(source_path_LR)

# HR_name_list.sort()
LR_name_list.sort()

for n in range(100):
    # HR_file_name = HR_name_list[n+800]
    LR_file_name = LR_name_list[n+800]

    # target_path_HR = os.path.join(root_path, 'DIV2K_test_tmp_HR', HR_file_name)
    target_path_LR = os.path.join(root_path, 'DIV2K_test_tmp_LR_bicubic/X4', LR_file_name)

    # source_path_HR_file = os.path.join(source_path_HR, HR_file_name)
    source_path_LR_file = os.path.join(source_path_LR, LR_file_name)

    # print(str(n)+source_path_HR_file+'-'+target_path_HR)
    # shutil.move(source_path_HR_file, target_path_HR)
    shutil.move(source_path_LR_file, target_path_LR)

