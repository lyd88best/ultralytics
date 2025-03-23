from ultralytics import YOLO

# Load a model
model = YOLO(r"./ultralytics/cfg/models/12/yolo12-CGRFPN.yaml")

# Train the model
train_results = model.train(
    data=r"./ultralytics/cfg/datasets/coco8.yaml",  # path to dataset YAML
    batch=32,  # batch size
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
# metrics = model.val()

# Perform object detection on an imageD:\pythonProjects\ultralytics\ultralytics\cfg\datasets\coco8\images\val
# results = model("path/to/image.jpg")
# results[0].show()

# Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model