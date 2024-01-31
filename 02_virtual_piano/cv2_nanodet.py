
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import cv2
import numpy as np

class CV2NanoDetONNX(object):
    STRIDES = (8, 16, 32)
    REG_MAX = 7
    PROJECT = np.arange(REG_MAX + 1)

    MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    MEAN = MEAN.reshape(1, 1, 3)
    STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    STD = STD.reshape(1, 1, 3)

    def __init__(
        self,
        model_path='nanodet_m.onnx',
        input_shape=320,
        class_score_th=0.35,
        nms_th=0.6,
    ):
        # 入力サイズ
        self.input_shape = (input_shape, input_shape)

        # 閾値
        self.class_score_th = class_score_th
        self.nms_th = nms_th

        # モデル読み込み        
        self.net = cv2.dnn.readNet('nanodet_finger_v2_sim.onnx')
        self.output_names = [
             "cls_pred_stride_8",
             "dis_pred_stride_8",
             "cls_pred_stride_16",
             "dis_pred_stride_16",
             "cls_pred_stride_32",
             "dis_pred_stride_32"
        ]

        # ストライド毎のグリッド点を算出
        self.grid_points = []
        for index in range(len(self.STRIDES)):
            grid_point = self._make_grid_point(
                (int(self.input_shape[0] / self.STRIDES[index]),
                 int(self.input_shape[1] / self.STRIDES[index])),
                self.STRIDES[index],
            )
            self.grid_points.append(grid_point)


    def inference(self, image):
        temp_image = copy.deepcopy(image)
        image_height, image_width = image.shape[0], image.shape[1]

        resize_image, new_height, new_width, top, left = self._resize_image(
            temp_image)
        blob = self._pre_process(resize_image)
        self.net.setInput(blob)
        preds = self.net.forward(self.output_names)

        #print(preds[0][0][:10])
        bboxes, scores, class_ids = self._post_process(preds)

        ratio_height = image_height / new_height
        ratio_width = image_width / new_width
        for i in range(bboxes.shape[0]):
            bboxes[i, 0] = max(int((bboxes[i, 0] - left) * ratio_width), 0)
            bboxes[i, 1] = max(int((bboxes[i, 1] - top) * ratio_height), 0)
            bboxes[i, 2] = min(
                int((bboxes[i, 2] - left) * ratio_width),
                image_width,
            )
            bboxes[i, 3] = min(
                int((bboxes[i, 3] - top) * ratio_height),
                image_height,
            )
        return bboxes, scores, class_ids

    def _make_grid_point(self, grid_size, stride):
        grid_height, grid_width = grid_size

        shift_x = np.arange(0, grid_width) * stride
        shift_y = np.arange(0, grid_height) * stride

        xv, yv = np.meshgrid(shift_x, shift_y)
        xv = xv.flatten()
        yv = yv.flatten()

        cx = xv + 0.5 * (stride - 1)
        cy = yv + 0.5 * (stride - 1)

        return np.stack((cx, cy), axis=-1)

    def _resize_image(self, image, keep_ratio=True):
        top, left = 0, 0
        new_height, new_width = self.input_shape[0], self.input_shape[1]

        if keep_ratio and image.shape[0] != image.shape[1]:
            hw_scale = image.shape[0] / image.shape[1]
            if hw_scale > 1:
                new_height = self.input_shape[0]
                new_width = int(self.input_shape[1] / hw_scale)

                resize_image = cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

                left = int((self.input_shape[1] - new_width) * 0.5)

                resize_image = cv2.copyMakeBorder(
                    resize_image,
                    0,
                    0,
                    left,
                    self.input_shape[1] - new_width - left,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
            else:
                new_height = int(self.input_shape[0] * hw_scale)
                new_width = self.input_shape[1]

                resize_image = cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

                top = int((self.input_shape[0] - new_height) * 0.5)

                resize_image = cv2.copyMakeBorder(
                    resize_image,
                    top,
                    self.input_shape[0] - new_height - top,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
        else:
            resize_image = cv2.resize(
                image,
                self.input_shape,
                interpolation=cv2.INTER_AREA,
            )

        return resize_image, new_height, new_width, top, left

    def _pre_process(self, image):
        image = image.astype(np.float32)
        image = (image - self.MEAN) / self.STD

        image = image.transpose(2, 0, 1).astype('float32')
        image = image.reshape(-1, 3, self.input_shape[0], self.input_shape[1])

        return image

    def _softmax(self, x, axis=1):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        #print(f"x_exp shape : {x_exp.shape}")
        #print(f"x_sum shape : {x_sum.shape}")

        s = x_exp / x_sum
        return s

    def _post_process(self, predict_results):
        class_scores = predict_results[::2]
        bbox_predicts = predict_results[1::2]
        bboxes, scores, class_ids = self._get_bboxes_single(
            class_scores,
            bbox_predicts,
            1,
            rescale=False,
        )

        return bboxes.astype(np.int32), scores, class_ids

    def _get_bboxes_single(
        self,
        class_scores,
        bbox_predicts,
        scale_factor,
        rescale=False,
        topk=1000,
    ):
        bboxes = []
        scores = []

        for stride, class_score, bbox_predict, grid_point in zip(
                self.STRIDES, class_scores, bbox_predicts, self.grid_points):
            if class_score.ndim == 3:
                class_score = class_score.squeeze(axis=0)
            if bbox_predict.ndim == 3:
                bbox_predict = bbox_predict.squeeze(axis=0)

            #print(bbox_predict.shape)
            #print(class_score.shape)
            bbox_predict = bbox_predict.reshape(-1, self.REG_MAX + 1)
            #print(bbox_predict.shape)

            bbox_predict = self._softmax(bbox_predict, axis=1)
            #print(f"after dot _softmax bbox_predict shape : {bbox_predict.shape}")
            bbox_predict = np.dot(bbox_predict, self.PROJECT).reshape(-1, 4)
            #print(f"after dot reshape bbox_predict shape : {bbox_predict.shape}")

            bbox_predict *= stride

            #print()



            if 0 < topk < class_score.shape[0]:
                max_scores = class_score.max(axis=1)
                topk_indexes = max_scores.argsort()[::-1][0:topk]


                grid_point = grid_point[topk_indexes, :]
                bbox_predict = bbox_predict[topk_indexes, :]
                class_score = class_score[topk_indexes, :]
                """
                print(f"class score shae : {class_score.shape}")
                print(class_score[:3])

                print("max score")
                print(max_scores.shape)
                print(max_scores[:3])
        
                print("topk_indexes")
                print(topk_indexes.shape)
                print(topk_indexes[:3])


                print("grid_point, bbox_predict, class score")



                print(grid_point.shape)
                print(bbox_predict.shape)
                print(class_score.shape)
                """
            #print()
            x1 = grid_point[:, 0] - bbox_predict[:, 0]
            y1 = grid_point[:, 1] - bbox_predict[:, 1]
            x2 = grid_point[:, 0] + bbox_predict[:, 2]
            y2 = grid_point[:, 1] + bbox_predict[:, 3]
            x1 = np.clip(x1, 0, self.input_shape[1])
            y1 = np.clip(y1, 0, self.input_shape[0])
            x2 = np.clip(x2, 0, self.input_shape[1])
            y2 = np.clip(y2, 0, self.input_shape[0])
            bbox = np.stack([x1, y1, x2, y2], axis=-1)

            bboxes.append(bbox)
            scores.append(class_score)

        bboxes = np.concatenate(bboxes, axis=0)
        if rescale:
            bboxes /= scale_factor
        scores = np.concatenate(scores, axis=0)

        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]
        class_ids = np.argmax(scores, axis=1)
        scores = np.max(scores, axis=1)

        indexes = cv2.dnn.NMSBoxes(
            bboxes_wh.tolist(),
            scores.tolist(),
            self.class_score_th,
            self.nms_th,
        )

        if len(indexes) > 0:
            bboxes = bboxes[indexes]
            scores = scores[indexes]
            class_ids = class_ids[indexes]
        else:
            bboxes = np.array([])
            scores = np.array([])
            class_ids = np.array([])

        return bboxes, scores, class_ids

# 살색 영역의 범위 지정 (HSV 색 공간)
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

def get_color_filtered_boxes(image):
    # 이미지를 HSV 색 공간으로 변환
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 살색 영역을 마스크로 만들기
    skin_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)
    # 모폴로지 연산을 위한 구조 요소 생성
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # 모폴로지 열림 연산 적용
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    # 마스크를 이용하여 살색 영역 추출
    skin_image = cv2.bitwise_and(image, image, mask=skin_mask)

    # 살색 영역에 대한 바운딩 박스 추출
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
        
    # 크기가 작은 박스와 큰 박스 제거
    color_boxes = []
    for (x, y, w, h) in bounding_boxes:
        if w * h > 100 * 100:
            # 약간 박스 더크게
            color_boxes.append((x - 10, y - 10, w + 20, h + 20))

    return color_boxes, skin_image



def draw_debug_roi(image, bboxes, scores, class_ids, x, y):
    #debug_image = copy.deepcopy(image)

    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        image = cv2.rectangle(
            image,
            (x1 + x, y1 + y),
            (x2 + x, y2 + y),
            (0, 255, 0),
            thickness=2,
        )

        score = '%.2f' % score
        text = '%s:%s' % (str(class_id), score)
        image = cv2.putText(
            image,
            text,
            (bbox[0] + x, bbox[1] - 10 + y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )
    
    return image

def draw_debug(image, elapsed_time, bboxes, scores, class_ids):
    debug_image = copy.deepcopy(image)

    for bbox, score, class_id in zip(bboxes, scores, class_ids):
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            thickness=2,
        )

        score = '%.2f' % score
        text = '%s:%s' % (str(class_id), score)
        debug_image = cv2.putText(
            debug_image,
            text,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            thickness=2,
        )

    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image
