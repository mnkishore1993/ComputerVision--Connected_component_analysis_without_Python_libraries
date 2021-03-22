"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments
    
    char_features = enrollment(characters)   
    
    
    
    labels,stats = detection(test_img)
    
    Rec_values = recognition(test_img,stats,char_features)
    
    final_chars = []
    
    for ele in Rec_values:
        if ele[0][2]>0 and ele[0][3]>0:
            final_chars.append({"bbox":list(ele[0]),"name":ele[1]})
    return final_chars


def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    char_features = {}
    
    for img in characters:        
        sift = cv2.SIFT_create()        
        kps,des = sift.detectAndCompute(img[1],None)
        if img[0] not in char_features.keys():
            char_features[img[0]] = (kps,des);
    return char_features

def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    
    #img_path = "/home/kishore/Desktop/Semester 2/computervision/projects/Project1/Project1/Project1/data/characters/test_img.jpg"
    #image = test_img

    label_img = np.zeros(test_img.shape)
    
    for i in range(0,test_img.shape[0]):
        for j in range(0,test_img.shape[1]):        
            label_img[i,j] = 0 if test_img[i,j]>150 else 1

    
    # =============================================================================
    #     First iteration 
    # =============================================================================
        
    current_label = 1
    UF_coll = {}
    for i in range(0,label_img.shape[0]):
        for j in range(0,label_img.shape[1]):
            if label_img[i,j]==0:
                continue;
            above_pixel =0 if i-1<0  else label_img[i-1,j]
            left_pixel = 0 if j-1<0 else label_img[i,j-1]
            
            if above_pixel==0 and left_pixel ==0:
                label_img[i,j] = current_label;            
                UF_coll[current_label] = [current_label];
                current_label = current_label +1;
                
            elif above_pixel != left_pixel:
                if above_pixel>0 and left_pixel>0:
                    label_img[i,j] = int(min(above_pixel,left_pixel));
                    UF_coll[int(max(above_pixel,left_pixel))].append(int(min(above_pixel,left_pixel)));
                    UF_coll[int(min(above_pixel,left_pixel))].append(int(max(above_pixel,left_pixel)));
                else:
                    label_img[i,j] = int(max(above_pixel,left_pixel));                       
              
            else :
                # case where above_pixel == left_pixel
                label_img[i,j] = above_pixel;
        
    
    # =============================================================================
    #     second iteration 
    # =============================================================================
    for i in range(0,label_img.shape[0]):
        for j in range(0,label_img.shape[1]):        
            label_img[i,j] = min(get_neighbourLabels(UF_coll,label_img[i,j],[])) 
    
    comps = sorted(list(map(lambda k : int(k),np.unique(label_img))))
    comps.remove(0)
    comp_stats = [(get_boundaries(label_img,x)) for x in comps if True]
    
    return label_img,comp_stats

def recognition(test_img,comp_stats,char_features):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    recognized_chars = []
    
    for i in range(0,len(comp_stats)):        
        boundaries = comp_stats[i]
        match_list = []
        for ele in char_features.keys():            
            match_list.append((ele,find_goods(test_img,boundaries,char_features[ele])))            
        
        match_list = list(filter(lambda x: x[1]> 0, match_list))        
        if len(match_list)>0:
            match_list = sorted(match_list,key=lambda k:k[1],reverse=True)
            recognized_chars.append((boundaries,match_list[0][0])) 
            
        else:
            recognized_chars.append((boundaries,"UNKNOWN"))
    
    return recognized_chars

def find_goods(test_img,boundaries,char_feature):    
    if boundaries[3]<=0 or boundaries[2]<=0:
        return 0
    img_comp = test_img[boundaries[1]:boundaries[1]+boundaries[3],boundaries[0]:boundaries[0]+boundaries[2]]
    sift1 = cv2.SIFT_create()
    sift2 = cv2.SIFT_create()
    key1,des1 = sift1.detectAndCompute(img_comp,None)
    
    des2 = char_feature[1]
    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        return len(good)
    else:
        return 0

def get_neighbourLabels(dict_lab,val,neighbours):
    output = neighbours
    #print(neighbours)
    if val not in dict_lab:
        #print("ere")
        return [val]
    else:
        
        for ele in list(set(dict_lab[val])):
            if ele not in neighbours:
                neighbours.append(ele)
                #print(ele)
                output.extend(get_neighbourLabels(dict_lab,ele,neighbours))
    return list(set(output));

def get_boundaries(img,comp):
    comp_idx = np.where(img==comp);
    if (len(comp_idx[1])>0) and len(comp_idx[0])>0:
        return int(min(comp_idx[1])),int(min(comp_idx[0])),int(int(max(comp_idx[1])) - int(min(comp_idx[1]))),int(int(max(comp_idx[0])) - int(min(comp_idx[0])));
    else:
        return 0,0,0,0;



def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = []
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(coordinates, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
