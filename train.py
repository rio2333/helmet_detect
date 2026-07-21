from ultralytics import YOLO


model = YOLO(r"E:\hatdetect\checkpoint\last.pt")


model.train(data=r"E:\hatdetect\data.yaml", epochs=50, imgsz=640, workers=1, project=r"E:\hatdetect\runs", name="hat_continue")


