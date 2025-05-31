from ultralytics import YOLO

model = YOLO('yolov8l.pt')

def main():
    model.train(data='Dataset/SplitData/dataoffline.yaml', epochs=30)
    print(f'trainnig is completed')


if __name__ == '__main__':
    main()