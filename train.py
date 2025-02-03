from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolo11m.pt")

    model.train(data = "dataset_custom.yaml", imgsz = 640, batch = 4, epochs = 100, workers = 1, device= 0)