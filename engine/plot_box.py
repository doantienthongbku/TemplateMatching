import cv2

def plot_box(d_img, boxes, nms_res=None):
    if nms_res is not None:
        color = (255, 0, 0)
        for i in nms_res:
            d_img = cv2.rectangle(d_img, boxes[i][0], boxes[i][1], color, 1)
    else:
        color = (0, 255, 0)
        for i in range(len(boxes)):
            d_img = cv2.rectangle(d_img, boxes[i][0], boxes[i][1], color, 1)

    return d_img