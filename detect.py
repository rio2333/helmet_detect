from ultralytics import YOLO

def main():
    # 加载权重
    model = YOLO("weights/best.pt")
    
    # 启动预测
    # source=0 代表默认摄像头，如果需要推理图片，改为图片路径即可
    model.predict(source=0, show=True)

if __name__ == "__main__":
    main()