import os
import glob
import shutil
from os import listdir
from pathlib import Path
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

"""
    The following script is to convert the PascalVOC 
    format into YOLO foramt


"""



def yolo_to_xml_bbox(bbox, w, h, cls):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax, cls]


def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]




# Classes
classes =  ['D00', 'D10', 'D20', 'D40']


# Counter
counter = 0


# Country folder
folder_name = "Norway"


# Direcotries path
root = "/cluster/work/salara/TDT17/datav2/data"
image_dir = f"/cluster/work/salara/TDT17/datav2/data/{folder_name}/train/images/"
output_dir = f"/cluster/work/salara/TDT17/datav2/data/{folder_name}/train/labels/"



# identify all the xml files in the annotations folder (input directory)
files = glob.glob(f"{root}/{folder_name}/train/annotations/xmls/*.xml")




# loop through each 
for fil in tqdm(files):
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    

    result = []


    # parse the content of the xml file
    tree = ET.parse(fil)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    for obj in root.findall('object'):
        label = obj.find("name").text

        # check for new classes and append to list
        if label not in classes:
            continue
        
        index = classes.index(label)
        pil_bbox = [int(float(x.text)) for x in obj.find("bndbox")]
        yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)


        # convert data to string
        bbox_string = " ".join([str(x) for x in yolo_bbox])
        result.append(f"{index} {bbox_string}")


    # DO NOT INCLUDE EMPTY LABELS: https://github.com/ultralytics/yolov3/issues/616
    if result:
        counter += 1
        # generate a YOLO format text file for each xml file
        with open(os.path.join(output_dir, f"{filename}.txt"), "w", encoding="utf-8") as f:
            #print("write")
            f.write("\n".join(result))

print(counter)