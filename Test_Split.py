from shutil import copytree, ignore_patterns, move, copy2
from random import sample
import os

def get_files(dir, src):
    '''Get list of files with path from given directory'''

    #Build file path according to dir passed

    #Below code retained temporarily incase needed later, originates from
    #first project idea which was abandoned.
    #if "lung" in dir:
    #    sub_src = os.path.join(src, 'lung_image_sets')
    #else:
    #    sub_src = os.path.join(src, 'colon_image_sets')

    sub_src = os.path.join(sub_src, dir)

    #Get all filenames with filepath
    files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk(sub_src) for f in filenames]

    return files

def main():
    '''A small script to set up test folders and randomly moves across'''

    #Copy directory structure with no files
    src = r'C:\Users\yblad\Documents\For Bsc\Year 3\AI\Assessed Work\Project\Code\original'
    dst = r'C:\Users\yblad\Documents\For Bsc\Year 3\AI\Assessed Work\Project\Code\original_test'

    copytree(src, dst, ignore=ignore_patterns('*.jpg', '*.jpeg'), dirs_exist_ok=False)

    #Define sub folders containing images then grab a list of each class's files
    dirs = ['Benign', 'Early', 'Pre', 'Pro']

    classes = []

    for i in range(len(dirs)):
        classes.append(get_files(dirs[i], src))

    #Check classes are expected length
    for i in range(len(classes)):
        assert len(classes[i]) == 5000

    #randomly sample 20% for test data from each class
    k = 1000

    test_data = []

    for i in range(len(classes)):
        test_data.append(sample(classes[i], k))

    files = []

    for data in test_data:
        for file in data:
            src = file

            #Generate destination path
            f_str = file.split("\\")
            for i in range(len(f_str)):
                if f_str[i] == "lung_colon_image_set_train":
                    f_str[i] = "lung_colon_image_set_test"

            dst = "\\".join(f_str)

            #Move file new location
            move(src, dst, copy_function=copy2)

    print("Done")

main()