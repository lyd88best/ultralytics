from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'D:\pythonProjects\ultralytics\yolov12n.pt')
    model.predict(source=r'D:\pythonProjects\ultralytics\ultralytics\assets'
                  , save=True
                  , show=False)