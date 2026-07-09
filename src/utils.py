import matplotlib.pyplot as plt
def visualize_sample(dataset, index):
    # Retrieve the sample
    image, target = dataset[index]

    # Display the image
    image_hwc = image.permute(1, 2, 0)  # Convert from CHW to HWC
    plt.imshow(image_hwc)
    plt.axis('off')

    # Draw bounding boxes. Box coords are normalized YOLO format (0-1),
    # so they need to be scaled to pixel dimensions before plotting.
    img_height, img_width = image_hwc.shape[0], image_hwc.shape[1]
    boxes = target["boxes"]
    for box in boxes:
        x_center, y_center, width, height = box
        x_center, width = x_center * img_width, width * img_width
        y_center, height = y_center * img_height, height * img_height
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        rect = plt.Rectangle(
            (x_min, y_min), width, height,
            fill=False, color='red', linewidth=2
        )
        plt.gca().add_patch(rect)

    plt.show()
