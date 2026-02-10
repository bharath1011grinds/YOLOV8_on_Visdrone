from ultralytics import YOLO
import torch
import sys

import os
data=os.path.join(os.path.dirname(__file__), '..', 'dataset', 'visdrone.yaml')

log_file = open('training_log.txt', 'w')
sys.stdout = log_file
sys.stderr = log_file



def train_visdrone():
    """
    Train YOLOv8 on VisDrone dataset
    """
    
    print("="*60)
    print("VisDrone Object Detection Training")
    print("="*60)
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = 0
    else:
        print("No GPU detected, using CPU (will be slow)")
        device = 'cpu'
    
    print("="*60)
    print()
    
    # Load pretrained YOLOv8 model
    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
    # Use 'n' for 4GB GPU
    model = YOLO('yolov8n.pt')
    
    print(f"Loaded pretrained YOLOv8-nano model")
    print()
    
    # Training configuration
    results = model.train(
        # Dataset
        data=data,
      
        # Training duration
        epochs=100,                    # Total epochs (can stop early with patience)
        patience=20,                   # Early stopping if no improvement for 20 epochs
        
        # Image settings
        imgsz=640,                     # Input image size (640x640)
        
        # Batch settings
        batch=8,                       # Batch size (your 4GB GPU can handle this)
                                       # If you get OOM errors, reduce to 4
        
        # Hardware
        device=device,                 # GPU device
        workers=4,                     # Data loading workers
        
        # Optimization
        optimizer='auto',              # Adam/SGD automatically chosen
        lr0=0.01,                      # Initial learning rate
        lrf=0.01,                      # Final learning rate (lr0 * lrf)
        momentum=0.937,                # SGD momentum
        weight_decay=0.0005,           # Weight decay
        warmup_epochs=3,               # Warmup epochs
        
        # Data augmentation (important for small objects!)
        hsv_h=0.015,                   # Hue augmentation
        hsv_s=0.7,                     # Saturation augmentation
        hsv_v=0.4,                     # Value augmentation
        degrees=0.0,                   # Rotation (+/- deg)
        translate=0.1,                 # Translation (+/- fraction)
        scale=0.5,                     # Scaling (+/- gain)
        shear=0.0,                     # Shear (+/- deg)
        perspective=0.0,               # Perspective (+/- fraction)
        flipud=0.0,                    # Flip up-down probability
        fliplr=0.5,                    # Flip left-right probability
        mosaic=1.0,                    # Mosaic augmentation probability
        mixup=0.0,                     # Mixup augmentation probability
        copy_paste=0.0,                # Copy-paste augmentation probability
        
        # Validation
        val=True,                      # Validate during training
        
        # Saving
        save=True,                     # Save checkpoints
        save_period=10,                # Save checkpoint every N epochs
        
        # Logging
        project='runs/train',          # Project directory
        name='visdrone_yolov8n',       # Experiment name
        exist_ok=False,                # Don't overwrite existing experiments
        
        # Visualization
        plots=True,                    # Save training plots
        
        # Other
        verbose=True,                  # Verbose output
        seed=42,                       # Random seed for reproducibility
        deterministic=True,            # Deterministic training
        single_cls=False,              # Multi-class training
        rect=False,                    # Rectangular training (keep False for detection)
        cos_lr=False,                  # Cosine learning rate scheduler
        close_mosaic=10,               # Disable mosaic in last N epochs
        amp=True,                      # Automatic Mixed Precision (faster training)
        fraction=1.0,                  # Use 100% of dataset (reduce for quick tests)
        profile=False,                 # Profile ONNX/TensorRT speeds
        freeze=None,                   # Freeze layers (None = train all)
    )
    
    print()
    print("="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved at: runs/train/visdrone_yolov8n/weights/best.pt")
    print(f"Results saved at: runs/train/visdrone_yolov8n/")
    print()
    
    # Print final metrics
    print("Final Metrics:")
    print(f"   mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"   mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print("="*60)


if __name__ == '__main__':
    train_visdrone()