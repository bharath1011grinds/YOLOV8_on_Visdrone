from ultralytics import YOLO


def main():

    conf = [0.10, 0.15, 0.20, 0.25, 0.30]

    model = YOLO('runs/detect/runs/train/visdrone_yolov8n/weights/best.pt')

    print("Conf | Precision | Recall | mAP50 | Detections")
    print("-" * 55)

    for c in conf:

        result = model.val(conf = c, verbose = False, data = 'dataset/visdrone.yaml', iou = 0.5)

        prec = result.box.mp
        rec = result.box.mr
        map50 = result.box.map50

        print(f"{c} | {prec:.3f} | {rec:.3f} | {map50:.3f} | Checking...")

    print("\nDone! Check which confidence gives best precision/recall balance.")


if __name__ == '__main__':
    main()