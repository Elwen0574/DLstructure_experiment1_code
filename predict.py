import argparse
from sre_parse import parse

from ultralytics import YOLO
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='./runs/detect/train/weights/best.pt')
    parser.add_argument('--image',type=str, default='./picture.jpg')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    model = YOLO(args.model)
    results = model.predict(args.image,iou=0.3)  # assumes `model` has been loaded
    results[0].show()
