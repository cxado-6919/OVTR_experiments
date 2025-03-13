import numpy as np
import random
import cv2
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer


def visualize_img_with_bbox(img_path, img, dt_instances: Instances, ref_pts=None, gt_boxes=None,category=False):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        dt_instances=list(dt_instances.values())[0]
        box=dt_instances["boxes"].cpu().numpy()
        scores=dt_instances["scores"].cpu().numpy()
        labels=dt_instances["labels"].cpu().numpy()
        img_show = draw_bboxes(img, np.concatenate(
            [box, scores.reshape(-1, 1), labels.reshape(-1, 1)],
            axis=-1),category=category)
        cv2.imwrite(img_path, img_show)

def draw_bboxes(ori_img, bbox, identities=None, mask=None, offset=(0, 0), cvt_color=False,category=False):
    img = ori_img
    for i, box in enumerate(bbox):
        if mask is not None and mask.shape[0] > 0:
            m = mask[i]
        else:
            m = None
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
            label = int(box[5])
        else:
            score = None
            label = None
        color = COLORS_10[i % len(COLORS_10)]

        label_str = category[label]
        if float(score)>0.4:
            img = plot_one_box([x1, y1, x2, y2], img, color, label_str, score=score, mask=m)
    return img

def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None, mask=None):
    tl = 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img, score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    # print("c1c2 = {} {}".format(c1, c2))
    if mask is not None:
        v = Visualizer(img, scale=1)
        vis_mask = v.draw_binary_mask(mask[0].cpu().numpy(), color="blue")
        img = vis_mask.get_image()
    return img

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238), (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47), (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144), (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128), (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238), (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154), (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128), (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220), (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]
