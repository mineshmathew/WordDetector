'''
input a folder where language wise sub folders are present
tthe sub folers has details of gt of word bounding boxes


output - 
average text line height for each language
average word width per language
average aspect ratio for each language

'''

import os, sys
import glob
input_dir = sys.argv[1]
split_name = sys.argv[2] #train, val or test
lang_folders = next (os.walk (input_dir))[1]

#stats_dict = {"language": None, "average_word_width":0, "average_text_height":0, "average_aspect_ratio":0, "average_line_width":None}

for lang_folder in lang_folders:
    stats_dict = {"language": None, "average_word_width":0, "average_text_height":0, "average_aspect_ratio":0, "average_line_width":None}
     
    width_sum = 0; height_sum = 0; aspect_ratio_sum =0
    stats_dict["language"] = lang_folder
    split_dir = os.path.join (input_dir, lang_folder, split_name)
    count = 0
    files_pattern = split_dir + "/*.words"
    if lang_folder == "urdu":
        files_pattern = split_dir + "/*.lines"
        stats_dict ["average_line_width"] = 0
        stats_dict ["average_word_width"] = None
       
    for file_name in glob.glob (files_pattern):
        
        #print (file_name)

        
        ann_f = open ( file_name, "r")
        
        
        while True:
            line1 = ann_f.readline().strip()
            line2 = ann_f.readline().strip()

            if not line1 or not line2 or len(line1) < 1 or len (line2) < 1:
                break
            line1_splitted = line1.split("\t")
            
           
            #print (line2_splitted)
            x1 = int (line1_splitted[-4])
            y1 = int (line1_splitted[-3])
            x2 = int (line1_splitted[-2])
            y2 = int (line1_splitted[-1])

            width = x2 - x1; width_sum += width
            height = y2 - y1; height_sum += height

            try:
                aspect_ratio = width/height; aspect_ratio_sum += aspect_ratio
            except:
                continue
            count += 1
    if lang_folder == "urdu":
        stats_dict ["average_line_width"] = width_sum / count
    else:
        stats_dict ["average_word_width"] = width_sum / count
    stats_dict ["average_text_height"] = height_sum/count
    stats_dict ["average_aspect_ratio"] =  aspect_ratio_sum/count
    print (stats_dict)
