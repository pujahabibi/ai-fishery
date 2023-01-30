from collections import defaultdict
from operator import itemgetter

def add_field_to_max_class(bounding_box_list):
    # Create a dictionary to store the count of each class
    class_counts = defaultdict(int)
    
    # Count the number of occurrences of each class
    for bbox in bounding_box_list:
        if bbox['class'] not in ['tanggal', 'blok']:
            class_counts[bbox['class']] += 1
    
    # Get the class with the maximum count
    max_class = max(class_counts, key=class_counts.get)
    max_count = class_counts[max_class]
    
    # Add a dummy bounding box with a class of 'None' to the bounding boxes with the maximum count, 
    # until they match the maximum count
    for bbox in bounding_box_list:
        if bbox['class'] == max_class and class_counts[bbox['class']] < max_count:
            class_counts[bbox['class']] += 1
            bounding_box_list.append({'class': 'None', 'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0})
    
    return class_counts

def convert_bounding_boxes(bounding_boxes):
    bounding_box_list = []
    for bb in bounding_boxes:
        xmin, ymin, xmax, ymax, class_, conf, text = bb
        bounding_box = {'class': class_, 'text':text, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bounding_box_list.append(bounding_box)
    return bounding_box_list

def sorted_bbox(bboxes):
    bounding_box_list = convert_bounding_boxes(bboxes)
    class_to_bounding_boxes = {}
    for bbox in bounding_box_list:
        class_name = bbox['class']
        if class_name not in class_to_bounding_boxes:
            class_to_bounding_boxes[class_name] = []
        class_to_bounding_boxes[class_name].append(bbox)

    sorted_bounding_boxes = []
    for bboxes in class_to_bounding_boxes.values():
        bboxes.sort(key=itemgetter('ymin'))
        sorted_bounding_boxes.extend(bboxes)

    return sorted_bounding_boxes

def grouping_bbox(bb):
    bboxes = sorted_bbox(bb)
    class_dict = {}
    single_entity = ['tanggal', 'blok']
    for i in range(len(bboxes)):
        if bboxes[i]['class'] in single_entity:
            class_dict[bboxes[i]['class']] = [bboxes[i]['text']]
        else:
            list_bbox = [bboxes[i]['text']]
            if bboxes[i]['class'] not in class_dict:
                class_dict[bboxes[i]['class']] = [list_bbox]
            else:
                class_dict[bboxes[i]['class']].append(list_bbox)

    return class_dict