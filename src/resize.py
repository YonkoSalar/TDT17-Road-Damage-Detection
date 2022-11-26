from PIL import Image, ImageDraw
import numpy as np
import albumentations as A
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
import cv2
from matplotlib import pyplot as plt
import matplotlib
import glob
from tqdm import tqdm

"""

    The following script was used to resize the images and bounding boxes in the Norwegian dataset.
    This was done because of the inconsistent image sies throughout the dataset.

"""



# Define classes
classes=['D00', 'D10', 'D20', 'D40']


def resize_image(img_arr, bboxes, h, w):
    """
    Get both the resize of image and bounding box


    Parameter:
        img_arr: original image as a numpy array
        bboxes: bboxes as numpy array where each row is 'x_min', 'y_min', 'x_max', 'y_max', "class_id"
        h: resized height dimension of image
        w: resized weight dimension of image
    
    return: 
        dictionary containing {image:transformed, bboxes:['x_min', 'y_min', 'x_max', 'y_max', "class_id"]}
    """
    # create resize transform pipeline
    transform = A.Compose(
        [A.Resize(height=h, width=w, always_apply=True)],
        bbox_params=A.BboxParams(format='pascal_voc'))

    transformed = transform(image=img_arr, bboxes=bboxes)

    return transformed





def read_xml(path_xml):
    """
    Retrieve information from xml file

    Parameter:
        path_xml: path to xml

    Return:
        bboxes: list of all bounding boxes for the xml file
        image: get image name for the corresponding image

    """
    

   
    # parse the content of the xml file
    tree = ET.parse(path_xml)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)
    image = str(root.find("filename").text)


    # Make list of all objects
    bboxes = np.zeros((len(root.findall('object')),5))

    for idx, obj in enumerate(root.findall('object')):
        
        label = obj.find("name").text

        # check for new classes and append to list
        if label not in classes:
            classes.append(label)
            continue 

        index = classes.index(label)
        pil_bbox = [int(float(x.text)) for x in obj.find("bndbox")]

        # add label
        index = classes.index(label)
        pil_bbox.append(index)

        bboxes[idx] = pil_bbox
    

    return bboxes, image




def draw_image(img, bboxes):
    """
    Function to review if the transformation was correct

    Parameter:
        img: image we want to draw
        bboxes: list of bounding boxes to the image

    Return:
        Saves image

    
    """

    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    
    for bbox in bboxes:
        bbox = list(bbox)
        draw.rectangle(bbox[:-1], outline="red", width=2)
        draw.text((bbox[2], bbox[1] - 10), str(classes[int(bbox[-1])]), fill=(255,0,0,255) )

    img.save("/cluster/work/salara/TDT17/draw_image.png")
    #img.show()


def rewrite_xml_image(img_path, xml_path, new_image, new_xml, dimension):
    
    # parse the content of the xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    root.find("size").find("width").text = str(dimension[1])
    root.find("size").find("height").text = str(dimension[0])

    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    
    for idx, obj in enumerate(root.findall('object')):
        
        for idy,x in enumerate(obj.find("bndbox")):
            x.text = str(int(new_xml[idx][idy]))
    
    tree.write(f'{xml_path}')

    # Overwrite image
    im = Image.fromarray(new_image)
    im.save(img_path)




# Read through all xml files in folders
xml_files = glob.glob("/cluster/work/salara/TDT17/datav2/data/Norway/train/annotations/xmls/*.xml")

# Counter
counter = 0

# Set Image dimension to resize 
new_dimensions = (1000, 1000)


# Read all xml files
for file in tqdm(xml_files):

    # Get the xml file
    xml_path = file
    
    # Get boundinx box and image name from xml
    bboxes, image = read_xml(xml_path)
    
    # Get image
    sample = Image.open(f"/cluster/work/salara/TDT17/datav2/data/Norway/train/images/{image}")
    sample_arr = np.asarray(sample)


    # Transform
    transformed_dict = resize_image(sample_arr, bboxes, new_dimensions[0], new_dimensions[1])


    # contains the image as array
    transformed_arr = transformed_dict["image"]

    # contains the resized bounding boxes
    transformed_info = np.array(list(map(list, transformed_dict["bboxes"]))).astype(float)
    
    # Rewrite xml and image 
    rewrite_xml_image(f"/cluster/work/salara/TDT17/datav2/data/Norway/train/images/{image}", xml_path, transformed_arr, transformed_info, new_dimensions)


    counter += 1

    # Draw image
    #draw_image(transformed_arr, transformed_info)
    
 

print("Number of train image resized: ", counter)





