import os
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

#Skip 0- ignored, 11 - others
visdrone_to_yolo ={
    1: 0,   # pedestrian
    2: 1,   # people
    3: 2,   # bicycle
    4: 3,   # car
    5: 4,   # van
    6: 5,   # truck
    7: 6,   # tricycle
    8: 7,   # awning-tricycle
    9: 8,   # bus
    10: 9,  # motor
}

def convert_visdrone_to_yolo(visdrone_root, output_root ,split='train'):

    images_dir = Path(visdrone_root) / 'images'
    annotations_dir = Path(visdrone_root) / 'annotations'
    
    output_images_dir = Path(output_root) / 'images' / split
    output_labels_dir = Path(output_root) / 'labels' / split
    
    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    annotation_files = sorted(annotations_dir.glob('*.txt'))
    print(f"\n{'='*50}")
    print(f"Converting {split} set: {len(annotation_files)} files")
    print(f"{'='*50}\n")
    
    skipped_count = 0
    converted_count = 0

    for ann_file in tqdm(annotation_files, desc= f"Converting{split} split"):
        
        image_name = ann_file.stem + '.jpg'
        image_path = Path(images_dir/image_name)

        if not image_path.exists():
            skipped_count +=1
            continue

        #Getting image dimensions 
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except:
            #print("error here")
            skipped_count+=1
            continue

        with open(ann_file,'r') as f:
            lines = f.readlines()

        #converting annotations
        yolo_annotations = []
        for line in lines:
            line = line.strip().split(',')
            if len(line)<6:
                continue   
            x,y,w,h = map(int, line[:4])
            label = int(line[5])

            if label not in visdrone_to_yolo:
                continue
            

            #converting to yolo format:
            centered_x, centered_y = (x+w/2)/img_width, (y+h/2)/img_height
            norm_width, norm_height = w/img_width, h/img_height
            yolo_label = visdrone_to_yolo[label]    

            yolo_annotations.append(f"{yolo_label} {centered_x:.6f} {centered_y:.6f} {norm_width:.6f} {norm_height:.6f}")

        #If there are no valid objects in the image, yolo_annotations is empty, skip the image.
        if not yolo_annotations:
            skipped_count+=1
            continue
        #copy image to new directory        
        shutil.copy(image_path, output_images_dir/image_name)

        #write the new annotations file to the output dir
        output_label_path = output_labels_dir/(ann_file.stem +'.txt')
        with open(output_label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))
        converted_count+=1

    print(f"\n Converted: {converted_count} images")
    print(f"Skipped: {skipped_count} images (no valid objects)")
    print(f"{'='*50}\n")


def create_yaml_config(output_root):
    """Create YAML config file for YOLOv8"""
    
    yaml_content = f"""# VisDrone Dataset Configuration
path: {Path(output_root).absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')

# Classes (10 valid classes, skipping 0 and 11)
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor

# Number of classes
nc: 10
"""
    
    yaml_path = Path(output_root) / 'visdrone.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created config file: {yaml_path}\n")

if __name__ =="__main__":
    VISDRONE_TRAIN = 'data/VisDrone2019-DET-train'
    VISDRONE_VAL = 'data/VisDrone2019-DET-val'
    OUTPUT_ROOT = 'dataset'

    #convert train set
    convert_visdrone_to_yolo(VISDRONE_TRAIN, output_root=OUTPUT_ROOT)
    
    #convert test set
    convert_visdrone_to_yolo(VISDRONE_VAL, OUTPUT_ROOT, split='val')

    #create the yaml config file
    create_yaml_config(output_root=OUTPUT_ROOT)

    print("Conversion complete! Ready to train YOLO.\n")

        






