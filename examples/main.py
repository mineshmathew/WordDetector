import argparse
from typing import List

import cv2
import matplotlib.pyplot as plt
from path import Path

from word_detector import detect, prepare_img, sort_multiline


def get_img_files(data_dir: Path) -> List[Path]:
    """Return all image files contained in a folder."""
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp', '*.tif']:
        res += Path(data_dir).files(ext)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('../data/line'))
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--sigma', type=float, default=11)
    parser.add_argument('--theta', type=float, default=7)
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--img_height', type=int, default=50)
    parsed = parser.parse_args()

    for fn_img in get_img_files(parsed.data):
        print(f'Processing file {fn_img}')

        # load image and process it
        img = prepare_img(cv2.imread(fn_img), parsed.img_height)
        detections = detect(img,
                            kernel_size=parsed.kernel_size,
                            sigma=parsed.sigma,
                            theta=parsed.theta,
                            min_area=parsed.min_area)

        # sort detections: cluster into lines, then sort each line
        lines = sort_multiline(detections)
    
         # plot results word wise

        ''' 
        plt.imshow(img, cmap='gray')
        num_colors = 7
        colors = plt.cm.get_cmap('rainbow', num_colors)
        for line_idx, line in enumerate(lines):
            for word_idx, det in enumerate(line):
                #import pdb; pdb.set_trace() 
                xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                plt.plot(xs, ys, c=colors(line_idx % num_colors))
                plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')

        plt.show()
        
        '''
        

        plt.imshow(img, cmap='gray')
        num_colors = 2
        colors = plt.cm.get_cmap('rainbow', num_colors)

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
                xs = [lowest_x, lowest_x, largest_x, largest_x, lowest_x]
                ys = [lowest_y, largest_y, largest_y, lowest_y, lowest_y]
                plt.plot(xs, ys, c=colors(line_idx % num_colors))
        plt.show()
if __name__ == '__main__':
    main()
