import xml.etree.ElementTree as ET
import os
import json

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
category_set2 = {'tanggal': 1,
 'blok': 2,
 'kolam': 3,
 'doc': 4,
 'jenis_pakan': 5,
 'f_d': 6,
 'fp_7': 7,
 'fp_11': 8,
 'fp_15': 9,
 'fp_20': 10,
 'ancho': 11,
 'do_pagi': 12,
 'do_sore_mlm': 13,
 'ph_pagi': 14,
 'ph_sore_mlm': 15,
 'suhu_pagi': 16,
 'suhu_sore_mlm': 17,
 'sal': 18,
 'mati': 19,
 't_air': 20,
 'kec': 21,
 'war': 22,
 'cuaca': 23,
 'resirkulasi': 24,
 'siphom': 25,
 'kincir': 26,
 'treatmen_air_pakan': 27}

image_set = set()

category_item_id = 0
image_id = 0
annotation_id = 0

def addCatItem(name, category_item_id):
    #global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)

    category_set[name] = category_item_id
    return category_item_id

def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id

def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    #left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    #right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    #right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)

def parseXmlFiles(xml_path): 
    for f in sorted(os.listdir(xml_path)):
        if not f.endswith('.xml'):
            continue
        
        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        print("FILE NAME", f)

        xml_file = os.path.join(xml_path, f)
        #print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        #elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None
            
            if elem.tag == 'folder':
                continue
            
            if elem.tag == 'filename':
                file_name = xml_path+elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')
                
            #add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    #print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name)) 
            #subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox ['xmin'] = None
                bndbox ['xmax'] = None
                bndbox ['ymin'] = None
                bndbox ['ymax'] = None
                
                current_sub = subelem.tag
                #print(current_sub)
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text

                    if " " in object_name:
                        object_name = object_name.replace(" ", "")
                    
                    # if "FLORIDINA ORANGE PET" in object_name:
                    #     print("REPLACE 35 ML with 60 ML")
                    #     object_name = 'FLORIDINA ORANGE PET 350'
                    # elif 'GRANITA CAPPUCINO FLAVORED WATER' in object_name:
                    #     print("REPLACE GRANITA")
                    #     object_name = 'GRANITA CAPPUCINO FLAVORED WATER 175'
                    # elif "Bear Brand Plain CAN 140" in object_name:
                    #     print("REPLACE Bear Brand Plain CAN 140")
                    #     object_name = 'Bear Brand Plain CAN 189'
                    # elif "The Botol Sosro Jasmine RGB 240" in object_name:
                    #     print("REPLACE The Botol Sosro Jasmine RGB 240")
                    #     object_name = 'The Botol Sosro Jasmine RGB 220'
                    # elif 'Teh Botol Sosro Jasmine RGB 220' in object_name:
                    #     print("REPLACE The Botol Sosro Jasmine RGB 240")
                    #     object_name = 'The Botol Sosro Jasmine RGB 220'
                    # elif "Uc 1000 Lemon RGB 120" in object_name:
                    #     print("REPLACE Uc 1000 Lemon RGB 120")
                    #     object_name = 'Uc 1000 Lemon RGB 140'
                    # elif "Uc 1000 Orange RGB 120" in object_name:
                    #     print("REPLACE Uc 1000 Orange RGB 120")
                    #     object_name = 'Uc 1000 Orange RGB 140'
                    # elif "S-tea Sosro Jasmine RGB 318" in object_name or "S-Tea Sosro Jasmine RGB 318" in object_name or "S-tee Sosro Jasmine RGB 318" in object_name:
                    #     print("REPLACE S-tea Sosro Jasmine RGB 318")
                    #     object_name = 'S-Tee Sosro Jasmine RGB 318'
                    # elif "Bintang Radle Lemon CAN 330" in object_name:
                    #     print("REPLACE Bintang Radle Lemon CAN 330")
                    #     object_name = 'Bintang Radler Lemon CAN 330'
                    # elif "Bintang Radle Lemon CAN 330" in object_name:
                    #     print("REPLACE Bintang Radle Lemon CAN 330")
                    #     object_name = 'Bintang Radler Lemon CAN 330'
                    # elif "FRESTEA Milk Tea Brown Sugar PET 330" in object_name:
                    #     print("REPLACE MILK TEA")
                    #     object_name = 'FRESTEA MILKTEA PET 330'
                    # elif "others" in object_name:
                    #     print("REPLACE OTHERS")
                    #     object_name = "foreign product"
                    # elif "RIO GULA BATU CUP" in object_name:
                    #     print("REPLACE RIO GULA BATU CUP")
                    #     object_name = "RIO GULA BATU CUP 200"
                    # elif "GREEN SANDS LEMON&GRAPE CAN" in object_name:
                    #     print("REPLACE RIO GULA BATU CUP")
                    #     object_name = "GREEN SANDS LEMON&GRAPE CAN 250"
                    # elif "GREEN SANDS LIME&LYCHEE CAN" in object_name:
                    #     print("REPLACE RIO GULA BATU CUP")
                    #     object_name = "GREEN SANDS LIME&LYCHEE CAN 250"
                    # elif "fanta sodawater pet 390" in object_name:
                    #     print("REPLACE fanta sodawater pet 390")
                    #     object_name = 'fanta_sodawater_pet_390'
                    # elif '\t' in object_name:
                    #     object_name = object_name.replace("\t", ' ')
                    #print(f)
                    #print("OBJECT_NAME", object_name)
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name, category_set2[object_name])
                        #print("NOT IN CT_SET", current_category_id)
                    else:
                        current_category_id = category_set[object_name]
                        #print("IN CT_SET", current_category_id)

                    #print(category_set)

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                #option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(float(option.text))

                #only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    print(bndbox['xmin'], bndbox['ymin'], bndbox['xmax'], bndbox['ymax'])
                    bbox = []
                    #x
                    bbox.append(bndbox['xmin'])
                    #y
                    bbox.append(bndbox['ymin'])
                    #w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    #h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    #print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id, bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox )

if __name__ == '__main__':
    xml_path = '/media/habibi/Data/projects/CenterNet/data/data_ocr/val/'
    json_file = 'instances_val.json'
    parseXmlFiles(xml_path)
    json.dump(coco, open(json_file, 'w'))