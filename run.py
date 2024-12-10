import argparse
from ultralytics import YOLO
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,default='yolov8s.pt')
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use (SGD/Adam)')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze during training')
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--hyp', type=str, default=None, help='Path to hyperparameter file')
    parser.add_argument('--augment', type=bool, default=False, help='Enable data augmentation')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    model = YOLO(args.model)
    results = model.train(data='./data.yaml', epochs=args.epochs, imgsz=args.size, batch=args.batch, workers=args.workers)
    metrics = model.val(data="data.yaml")  # assumes `model` has been loaded
    print('map: ', metrics.box.map)  # mAP50-95
    print('map50: ', metrics.box.map50)  # mAP50
    print('map75: ', metrics.box.map75)  # mAP75
    print('maps: (每个类别的平均精度)', metrics.box.maps)