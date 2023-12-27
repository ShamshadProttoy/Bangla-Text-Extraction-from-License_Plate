from ultralytics import YOLO

# Load the YOLO model
model = YOLO('platedata.pt')

# Perform object detection on the image
results = model('plate.jpg')

# Access class labels from the YOLO model
class_labels = model.names

class_labels_sorted = []

for result in results:
    boxes = result.boxes.xyxy
    x_center = (boxes[:, 0] + boxes[:, 2]) / 2

    # Separate detections into numerical and non-numerical
    numerical_detections = []
    non_numerical_detections = []
    
    for x, class_id in zip(x_center.tolist(), result.boxes.cls.cpu().numpy()):
        label = class_labels[int(class_id)]
        if label.isnumeric():
            numerical_detections.append((x, label))
        else:
            non_numerical_detections.append((x, label))

    # Sort each group
    sorted_numerical = sorted(numerical_detections, key=lambda x: x[0])
    sorted_non_numerical = sorted(non_numerical_detections, key=lambda x: x[0])

    # Combine and extract labels
    sorted_combined = sorted_numerical + sorted_non_numerical
    sorted_class_labels = [label for _, label in sorted_combined]
    
    class_labels_sorted.append(sorted_class_labels)

# Print sorted class labels
print(class_labels_sorted)