import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

#path to the model's best trained params
model = YOLO('runs/detect/runs/train/visdrone_yolov8n/weights/best.pt')

CLASS_NAMES = {
            0 : 'pedestrian',
            1 : 'people',
            2 : 'bicycle',
            3 : 'car',
            4 : 'van',
            5 : 'truck',
            6 : 'tricycle',
            7 : 'awning-tricycle',
            8 : 'bus',
            9 : 'motor'
}


def detect_objects(image, conf_threshold, iou_threshold):
    """ ARGS: 
        Run object detection on uploaded image.
        image : PIL Image or Np array
        conf_threshold : minimum confidence threshold for an obj to be detected and classified(combined confidence)
        iou_threshold : iou threshold for NMS - suppresses the predictions with lesser confidence if iou>iou_threshold
        
        RETURNS: 
        annotated_image : Image with bounding boxes
        detection_summary : Text summary of detections
        
    """

    result = model.predict(image, conf = conf_threshold, iou = iou_threshold, verbose = False)

    #Get the annotated image
    ann_image = result[0].plot() #Returns the image as an np array with the boxes drawn
    ann_image = Image.fromarray(ann_image)#converts the array to image

    class_counts = {}
    boxes = result[0].boxes

    for box in boxes: 
        cls = int(box.cls[0])
        cls_name = CLASS_NAMES.get(cls, f'class_{cls}')

        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    #Creating summary text
    total_detections = len(boxes)
    summary = f"**Total Detections: {total_detections}**\n\n"

    if class_counts: 
        for name, count in sorted(class_counts.items(), key = lambda x : x[1], reverse = True):

            summary+= f"-{name} : {count}\n" 
    else: 

        summary+= "No objects detected, try lowering the confidence threshold."

    return ann_image, summary

with gr.Blocks(title= "Visdrone Object detection") as demo:

    gr.Markdown(
        """
        # VisDrone Object Detection with YOLOv8
        
        Upload a drone/aerial image to detect vehicles, pedestrians, and other objects.
        
        **Model:** YOLOv8-nano trained on VisDrone2024 dataset  
        **Performance:** 43.5% mAP@50, 56.3% precision @ conf=0.25
        """
    )

    with gr.Row():
        with gr.Column():
            #Input image
            img = gr.Image(type='pil', label= 'Upload Drone Image', height = 400)



            #Controls
            confidence_slider = gr.Slider(minimum=0.05, maximum=0.95, value=0.25, step=0.05,
                                          label= 'Confidence Threshold', 
                                          info= "Lower = more detections (but more false positives)")
            
            iou_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.45, step =0.05,
                                   label= 'IoU Threshold',
                                   info= 'Lower IoU - fewer overlapping boxes, might miss out objects if too low. Spam predictions if too high.')
            
            detect_button = gr.Button("Detect Objects", variant='primary')

        with gr.Column():
            #Output Image
            output_image = gr.Image(label='Detection Results', height=400)

            #Output Summary
            output_summary = gr.Markdown(label='Detection Summary')
        
    #Examples section
    gr.Markdown("### Try Example Images")
    gr.Examples(
        examples=[
            ["dataset/images/val/0000001_03999_d_0000007.jpg", 0.25, 0.45],
            ["dataset/images/val/0000001_05499_d_0000010.jpg", 0.25, 0.45],
            ["dataset/images/val/0000022_00000_d_0000004.jpg", 0.20, 0.45],
        ],
        inputs=[img, confidence_slider, iou_slider],
        outputs=[output_image, output_summary],
        fn=detect_objects,
        cache_examples=False,
    )
    # Model info footer
    gr.Markdown(
        """
        ---
        **About this model:**
        - Trained on 6,471 drone images from VisDrone2019 dataset
        - Detects 10 object classes (vehicles, pedestrians, bicycles, etc.)
        - Key challenge: Small objects (pedestrians <20 pixels)
        - Best performance on vehicles (cars: 74% accuracy)
        
        **Tips:**
        - Use conf=0.25 for high precision (fewer false positives)
        - Use conf=0.15 for high recall (detect more objects)
        - Urban/crowded scenes work best
        """
    )

    #connecting button to function
    detect_button.click(fn = detect_objects,
                        inputs=[img, confidence_slider, iou_slider], 
                        outputs= [output_image, output_summary])
    
if __name__ =='__main__':
    demo.launch(
        share=True,  # Creates public link for sharing
        server_name="0.0.0.0",  # Accessible from network
        server_port=7860
    )


            
    

