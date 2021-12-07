import argparse
from typing import List

import numpy as np
from nms import non_max_suppression_slow, non_max_suppression_fast
import cv2
import matplotlib.pyplot as plt
from path import Path
import os
from word_detector import detect, prepare_img, sort_multiline
from stats import average_aspect_ratios, average_text_heights


def get_img_files(data_dir: Path) -> List[Path]:
    """Return all image files contained in a folder."""
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp', '*.tif']:
        res += Path(data_dir).files(ext)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True )
    parser.add_argument('--language', required=True )
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--sigma', type=float, default=11)
    #parser.add_argument('--theta', type=float, default=7)
    parser.add_argument('--min_area', type=int, default=100)
    #parser.add_argument('--img_height', type=int, default=50)
    parsed = parser.parse_args()

    for fn_img in get_img_files(parsed.data):
        print(f'Processing file {fn_img}')

        # load image and process it
        img = cv2.imread(fn_img)

        #resize page image to a new height so that text lines are of around 30 pixels in height
        height, width = img.shape[:2]
        #print ("original height is ", height)
        average_text_height_for_language = average_text_heights[parsed.language]

        
        img_height = int (height *   30 / average_text_height_for_language )

        theta = average_aspect_ratios [parsed.language]
        #import pdb; pdb.set_trace()
        #print ("new height is ", img_height)
        img = prepare_img(img, img_height)
        detections = detect(img,
                            kernel_size=parsed.kernel_size,
                            sigma=parsed.sigma,
                            theta=theta,
                            min_area=parsed.min_area)

        #import pdb; pdb.set_trace() 

        ##----------------------------NMS on word bbs--------------------------------------------
        # lets get the bounding boxes as an numpy array to use with nms code
        word_bbs = []
        
        for det  in detections:
            bounding_box = [det.bbox.x, det.bbox.y, det.bbox.x + det.bbox.w , det.bbox.y + det.bbox.h]
            word_bbs.append (bounding_box)

        word_boxes = np.array (word_bbs)
        idxs,_ = non_max_suppression_fast(word_boxes, probs=None, overlapThresh=0.1)

        # now from detections we need to pick only those  indices in the idxs

        detections2 = [detections[i] for i in idxs ]
        #import pdb; pdb.set_trace()

        # sort detections: cluster into lines, then sort each line
        lines = sort_multiline(detections2)
    
        
        line_bbs = []

        for line_idx, line in enumerate(lines):
            lowest_x = 100000
            largest_x = -1
            lowest_y = 100000
            largest_y = -1
            words_count = 0
            for word_idx, det in enumerate (line):
                words_count += 1
                x1 = det.bbox.x
                y1 = det.bbox.y

                x2 = x1 + det.bbox.w
                y2 = y1 + det.bbox.h

            
                if x1 < lowest_x :
                    lowest_x = x1

                if x2 > largest_x :
                    largest_x = x2
                if y1 < lowest_y:
                    lowest_y = y1
                if y2 > largest_y:
                    largest_y = y2
            if words_count > 0:
                line_bbs.append ([lowest_x, lowest_y, largest_x, largest_y])
        
        
        
        ##----------------------------NMS on line boxes ---------------------------------

        line_boxes =  np.array (line_bbs)
        _, boxes_after_nms = non_max_suppression_fast(line_boxes, probs=None, overlapThresh=0.05)
        
        
        ## ------------------saving boxes ---------------------------------------------------
        file_name_pattern = "_scale_space.lines"
        image_name = os.path.basename (fn_img)
        base_name = os.path.splitext (image_name)[0]
        lines_file = os.path.join (parsed.data, base_name + file_name_pattern)
        
        #import pdb; pdb.set_trace() 

        line_file_string = ""
        for i, each_line in enumerate (boxes_after_nms):
            count = i + 1
            values_to_write = [count, each_line[0], each_line[1], each_line[2], each_line[3]]
            line_str = "\t".join ([str(v) for v in values_to_write])
            line_file_string += line_str
            line_file_string += "\n"
        line_file_string = line_file_string.strip()

        #import pdb; pdb.set_trace()
        with open (lines_file, "w") as f:
            f.write (line_file_string)





        
        


if __name__ == '__main__':
    main()
