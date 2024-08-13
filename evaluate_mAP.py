import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, detect_image, image_preprocess, postprocess_boxes, nms, read_class_names
from yolov3.configs import *
import shutil
import json
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: print("RuntimeError in tf.config.experimental.list_physical_devices('GPU')")

def voc_ap(rec, prec):
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

def get_mAP(Yolo, dataset, score_threshold=0.25, iou_threshold=0.50, TEST_INPUT_SIZE=TEST_INPUT_SIZE):
    MINOVERLAP = 0.5
    NUM_CLASS = read_class_names(TRAIN_CLASSES)

    ground_truth_dir_path = 'mAP/ground-truth'
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)

    if not os.path.exists('mAP'): os.mkdir('mAP')
    os.mkdir(ground_truth_dir_path)

    print(f'\ncalculating mAP{int(iou_threshold*100)}...\n')

    gt_counter_per_class = {}
    for index in range(dataset.num_samples):
        ann_dataset = dataset.annotations[index]
        original_image, bbox_data_gt = dataset.parse_annotation(ann_dataset, True)

        if len(bbox_data_gt) == 0:
            bboxes_gt = []
            classes_gt = []
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        ground_truth_path = os.path.join(ground_truth_dir_path, str(index) + '.txt')
        num_bbox_gt = len(bboxes_gt)

        bounding_boxes = []
        for i in range(num_bbox_gt):
            class_name = NUM_CLASS.get(int(classes_gt[i]), "Unknown")
            xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
            bbox = xmin + " " + ymin + " " + xmax + " " + ymax
            bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})

            if class_name in gt_counter_per_class:
                gt_counter_per_class[class_name] += 1
            else:
                gt_counter_per_class[class_name] = 1
        with open(f'{ground_truth_dir_path}/{str(index)}_ground_truth.json', 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    times = []
    json_pred = [[] for _ in range(n_classes)]
    for index in range(dataset.num_samples):
        ann_dataset = dataset.annotations[index]
        original_image, bbox_data_gt = dataset.parse_annotation(ann_dataset, True)

        image = image_preprocess(np.copy(original_image), [TEST_INPUT_SIZE, TEST_INPUT_SIZE])
        image_data = image[np.newaxis, ...].astype(np.float32)

        t1 = time.time()
        if YOLO_FRAMEWORK == "tf":
            if tf.__version__ > '2.4.0':
                pred_bbox = Yolo(image_data)
            else:
                pred_bbox = Yolo.predict(image_data)
        elif YOLO_FRAMEWORK == "trt":
            batched_input = tf.constant(image_data)
            result = Yolo(batched_input)
            pred_bbox = [value.numpy() for value in result.values()]
        
        t2 = time.time()
        
        times.append(t2 - t1)
        
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_image, TEST_INPUT_SIZE, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        gt_classes_found = [NUM_CLASS.get(int(cls), "Unknown") for cls in bbox_data_gt[:, 4]] if len(bbox_data_gt) > 0 else []
        pred_classes_found = [NUM_CLASS.get(int(bbox[5]), "Unknown") for bbox in bboxes] if len(bboxes) > 0 else []

        print(f"Image index: {index}")
        print(f"Ground Truth Classes: {gt_classes_found}")
        print(f"Predicted Classes: {pred_classes_found}")

        # 추가된 코드: 모든 예측 박스를 출력
        for bbox in bboxes:
            print(f"Predicted bbox: {bbox}")
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])
            class_name = NUM_CLASS.get(class_ind, "Unknown")
            print(f"Class: {class_name}, Score: {score}")
            
            if class_name == "Unknown":
                continue
                
            score = '%.4f' % score
            xmin, ymin, xmax, ymax = list(map(str, coor))
            bbox = xmin + " " + ymin + " " + xmax + " " + ymax
            json_pred[gt_classes.index(class_name)].append({"confidence": str(score), "file_id": str(index), "bbox": str(bbox)})

    ms = sum(times) / len(times) * 1000
    fps = 1000 / ms

    for class_name in gt_classes:
        json_pred[gt_classes.index(class_name)].sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(f'{ground_truth_dir_path}/{class_name}_predictions.json', 'w') as outfile:
            json.dump(json_pred[gt_classes.index(class_name)], outfile)

    sum_AP = 0.0
    ap_dictionary = {}
    with open("mAP/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            predictions_file = f'{ground_truth_dir_path}/{class_name}_predictions.json'
            predictions_data = json.load(open(predictions_file))

            nd = len(predictions_data)
            tp = [0] * nd
            fp = [0] * nd
            for idx, prediction in enumerate(predictions_data):
                file_id = prediction["file_id"]
                gt_file = f'{ground_truth_dir_path}/{str(file_id)}_ground_truth.json'
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                bb = [float(x) for x in prediction["bbox"].split()]
                for obj in ground_truth_data:
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                if ovmax >= MINOVERLAP:
                    if not gt_match["used"]:
                        tp[idx] = 1
                        gt_match["used"] = True
                        count_true_positives[class_name] += 1
                        with open(gt_file, 'w') as f:
                            json.dump(ground_truth_data, f)
                    else:
                        fp[idx] = 1
                else:
                    fp[idx] = 1

            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            rec = [float(tp[idx]) / gt_counter_per_class[class_name] for idx in range(len(tp))]
            prec = [float(tp[idx]) / (fp[idx] + tp[idx]) for idx in range(len(tp))]

            ap, mrec, mprec = voc_ap(rec, prec)
            sum_AP += ap
            text = "{0:.3f}%".format(ap * 100) + " = " + class_name + " AP  "

            rounded_prec = ['%.3f' % elem for elem in prec]
            rounded_rec = ['%.3f' % elem for elem in rec]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")

            print(text)
            ap_dictionary[class_name] = ap

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes

        text = "mAP = {:.3f}%, {:.2f} FPS".format(mAP * 100, fps)
        results_file.write(text + "\n")
        print(text)
        
        return mAP * 100

if __name__ == '__main__':
    if YOLO_FRAMEWORK == "tf":
        if YOLO_TYPE == "yolov4":
            Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
        if YOLO_TYPE == "yolov3":
            Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

        if YOLO_CUSTOM_WEIGHTS == False:
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
            load_yolo_weights(yolo, Darknet_weights)
            
            # 2번: 가중치 로드를 생략한 후 첫 번째 레이어 가중치 출력
            print("First layer weights after skipping weight loading:", yolo.layers[0].get_weights())
            
            # 6번: 모든 레이어에서 가중치 로드 여부 확인
            for i, layer in enumerate(yolo.layers):
                print(f"Layer {i} ({layer.name}):")
                weights = layer.get_weights()
                if weights:
                    print("Weights loaded successfully.")
                else:
                    print("No weights loaded.")
        
        else:
            yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
            yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}")

            # 위와 동일한 디버깅 코드 추가 가능
    elif YOLO_FRAMEWORK == "trt":
        # TensorRT 관련 코드 생략
        pass

    testset = Dataset('test', TEST_INPUT_SIZE=YOLO_INPUT_SIZE)
    get_mAP(yolo, testset, score_threshold=0.05, iou_threshold=0.50, TEST_INPUT_SIZE=YOLO_INPUT_SIZE)
