#encoding=utf-8
import sys
import argparse
from yolo import YOLO
from PIL import Image
import time
import cv2
import pickle
import numpy as np

def save_gt_pkl(test_path):
    '''将GT序列化'''
    with open(test_path) as f:
        lines = f.readlines()

    all_boxes_gt = [[[] for _ in range(len(lines))]  #all_boxes [cls_num, image_num]
                    for _ in range(10)]

    # 获取某一类别的所有框，ground-true
    for i,line in enumerate(lines): #i表示第i张图像
        for tar in line.split()[1:]:
            bb=tar.split(",")
            cls_=int(bb[-1])
            bb_=[float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
            all_boxes_gt[cls_][i].append(bb_)

    gt_file="output/gt.pkl"
    with open(gt_file, 'wb') as f:
        pickle.dump(all_boxes_gt, f, pickle.HIGHEST_PROTOCOL)


def save_test_pkl(test_path):
    '''对测试集进行检测，将结果序列化'''
    with open(test_path) as f:
        lines = f.readlines()

    all_boxes = [[[] for _ in range(len(lines))]  #all_boxes [cls_num, image_num]
                    for _ in range(10)]

    tic=time.time()
    yolo=YOLO() #init YOLO class
    toc=time.time()
    print("init time:",toc-tic)

    for i,line in enumerate(lines): #i表示第i张图像
        im_path=line.split()[0]
        im=Image.open(im_path)

        tic=time.time()
        out_boxes,out_scores,out_classes=yolo.detect_image_result(im)
        toc=time.time()
        print("detection time:",toc-tic)

        for j,cls_ in enumerate(out_classes):
            # 注意测试结果中的box坐标顺序
            tmp=[out_boxes[j][1], out_boxes[j][0], out_boxes[j][3], out_boxes[j][2]] #x_min y_min x_max y_max
            tmp.append(out_scores[j])
            all_boxes[cls_][i].append(tmp)

    print("统计检测到的类别个数")
    for j in range(10): #10类
        cls_num=0
        for tmp in all_boxes[j]:
            cls_num=cls_num+len(tmp)
        print("第 {} 类：{}".format(j, cls_num))

    det_file="output/detections.pkl"
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    

#计算AP
def calc_ap():
    '''计算平均精度AP，mAP；通过读取GT和检测结果，计算IOU大于0.5以上的检测框；目前没有对检测结果做非极大值抑制'''
    ovthresh=0.5
    all_boxes=pickle.load(open("output/detections.pkl",'rb'))
    all_boxes_gt=pickle.load(open("output/gt.pkl","rb"))

    print("统计检测到的类别个数")
    for j in range(10): #10类
        cls_num=0
        for tmp in all_boxes[j]:
            cls_num=cls_num+len(tmp)
        print("第 {} 类：{}".format(j, cls_num))
    
    print("统计GT的类别个数")
    for j in range(10): #10类
        cls_num=0
        for tmp in all_boxes_gt[j]:
            cls_num=cls_num+len(tmp)
        print("第 {} 类：{}".format(j, cls_num))

    ap_all=[]
    for i in range(10):
        tp=[]
        fp=[]
        npos = 0 #某类别的所有正例数量，用来计算召回率
        for j in range(len(all_boxes[i])):
            BBGT=np.array(all_boxes_gt[i][j]) #BBGT表示检测图像中的所有ground-true框
            npos=npos + BBGT.shape[0]
            for bb in np.array(all_boxes[i][j]): #bb表示检测图像中的一个检测框
                ovmax=-np.inf
                if BBGT.size >0:
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                    
                    overlaps = inters / uni  #IOU
                    ovmax = np.max(overlaps)  #bb表示测试集中某一个检测出来的框的四个坐标，BBGT表示和bb同一图像上的所有检测框，取其中IOU最大的作为检测框的ground-true
                    # jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    tp.append(1) #预测为正，实际为正
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1) #预测为正，实际为负

        fp=np.array(fp)
        tp=np.array(tp)
        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos) 
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        #计算ap
        ap=0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec>=t)==0:
                p=0
            else:
                p=np.max(prec[rec>=t])
            ap=ap+p/11.
        ap_all.append(ap)
        print("第 {} 类 AP:{}".format(i, ap))
    # mAP
    ap_all=np.array(ap_all)
    print("mAP",ap_all.mean())

if __name__=="__main__":
    calc_ap()
    # test_path="NWPU_train.txt"
    # save_test_pkl(test_path)
    print("")

