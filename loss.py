#ciou and diou loss python implementation
import math
pi = math.pi
atan = math.atan

class boxabs:
    left, right, top, bot = 0, 0, 0, 0

class box:
    x, y, w, h = 0, 0, 0, 0

def overlap(x1, w1, x2, w2):
    l1 = x1 - w1/2
    l2 = x2 - w2/2
    left = l1 if l1 - l2 > 0 else l2
    r1 = x1 + w1/2
    r2 = x2 + w2/2
    right = r1 if r1 - r2 < 0 else r2
    return right - left

def box_intersection(a, b):
    """
    args:
     a type:box
     b type:box 
    """
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if(w < 0 or h < 0):
        return 0
    area = w*h;
    return area

def box_union(a, b):
    """
    args:
     a type:box
     b type:box 
    """
    i = box_intersection(a, b)
    u = a.w*a.h + b.w*b.h - i
    return u

def box_c(a, b):
    """
    arg: two boxes a b type: box
    return: smallest box that fully encompases a and b
    """
    ba = boxabs()
    ba.top = min(a.y - a.h / 2, b.y - b.h / 2)
    ba.bot = max(a.y + a.h / 2, b.y + b.h / 2)
    ba.left = min(a.x - a.w / 2, b.x - b.w / 2)
    ba.right = max(a.x + a.w / 2, b.x + b.w / 2)
    return ba

def box_iou(a, b):
    """
    args:
     a type:box
     b type:box 
    """
    I = box_intersection(a, b)
    U = box_union(a, b)
    if (I == 0 or U == 0):
        return 0
    return I / U

def box_ciou(pred_box, gtbox):
    ba = box_c(pred_box, gtbox)
    w = ba.right - ba.left
    h = ba.bot - ba.top
    #Diagonal distance of ba
    c = w * w + h * h
    iou = box_iou(pred_box, gtbox)
    # w = 0. h = 0
    if c == 0:
        return iou
    #center point distance
    u = (pred_box.x - gtbox.x) * (pred_box.x - gtbox.x) + (pred_box.y - gtbox.y) * (pred_box.y - gtbox.y)
    d = u / c
    ar_gt = gtbox.w / gtbox.h
    ar_pred = pred_box.w / pred_box.h
    ar_loss = 4 / (pi * pi) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred))
    alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
    ciou_term = d + alpha * ar_loss
    return iou - ciou_term


def box_diou(pred_box, gtbox):
    ba = box_c(pred_box, gtbox)
    w = ba.right - ba.left
    h = ba.bot - ba.top
    c = w * w + h * h;
    iou = box_iou(pred_box, gtbox)
    if (c == 0):
        return iou
    d = (pred_box.x - gtbox.x) * (pred_box.x - gtbox.x) + (pred_box.y - gtbox.y) * (pred_box.y - gtbox.y)
    u = math.pow(d / c, 0.6)
    diou_term = u

    return iou - diou_term

if __name__ == "__main__":
    pred_box = box()
    pred_box.x, pred_box.y, pred_box.w, pred_box.h = 0.4, 0.6, 0.3, 0.2
    gtbox = box()
    gtbox.x, gtbox.y, gtbox.w, gtbox.h = 0.5, 0.5, 0.4, 0.3
    print("diou loss:", box_diou(pred_box, gtbox))
    print("ciou loss:", box_ciou(pred_box, gtbox))
    