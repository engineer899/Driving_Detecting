import torch

import numpy as np

from utils1.datasets import letterbox
from utils1.general import non_max_suppression, scale_coords, xyxy2xywh


def detect(model, frame, stride, imgsz):
    # print(type(frame))

    result = []
    # dataset = LoadImages(frame, img_size=imgsz, stride=stride)

    img = letterbox(frame, imgsz, stride=stride)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)

    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img)
    img = img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # img = torch.from_numpy(img).to('cuda')
    pred = model(img, augment=False)
    pred =pred[0]


    # print(pred)
    # print(pred.size())
    pred = non_max_suppression(pred, 0.5, 0.45,  agnostic=False)
    # print(pred)

    for i, det in enumerate(pred):
        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                result.append([int(cls), xywh])

    return result

def detect_onnx(ort_session, frame, stride, imgsz):

    result = []
    # dataset = LoadImages(frame, img_size=imgsz, stride=stride)
    img = letterbox(frame, imgsz, stride=stride)[0]

    img = img[:, :, ::-1].transpose(2, 0, 1)

    img = np.ascontiguousarray(img)  # 将内存变为连续
    img = img.astype(np.float32)

    h = img.shape[1]
    w = img.shape[2]
    new_img = np.zeros([3, 640, 640])
    # imgs = Image.fromarray(img)
    # imgs.show()  # 显示图片
    # imgs = Image.fromarray(new_img[-1,0,1])
    # imgs.show()  # 显示图片
    new_img[:, :h, :w] = img
    img = new_img

    img /= 255.0
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)



    outputs = [x.name for x in ort_session.get_outputs()]
    input_name = ort_session.get_inputs()[0].name
    onnx_result = ort_session.run(outputs, {input_name: img})
    onnx_result =torch.from_numpy(onnx_result[0])

    pred = non_max_suppression(onnx_result, 0.5, 0.45,  agnostic=False)

    for i, det in enumerate(pred):
        gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                result.append([int(cls), xywh])

    return result


# def detect_onnx(ort_session, frame, stride, imgsz):
#
#     result = []
#     # dataset = LoadImages(frame, img_size=imgsz, stride=stride)
#     img = letterbox(frame, imgsz, stride=stride)[0]
#     img = img[:, :, ::-1].transpose(2, 0, 1)
#
#     img = np.ascontiguousarray(img) # 将内存变为连续
#     img = img.astype(np.float32)
#
#     # h = img.shape[1]
#     # w = img.shape[2]
#     # new_img = np.zeros([3, 640, 640])
#     # new_img[:, :h, :w] = img
#     # img = new_img
#
#     # showimg(img)
#
#     img /= 255
#     if len(img.shape) == 3:
#         img = np.expand_dims(img, axis=0)
#     # img = img.astype(np.float32)
#
#     outputs = [x.name for x in ort_session.get_outputs()]
#     input_name = ort_session.get_inputs()[0].name
#     onnx_result = ort_session.run(outputs, {input_name: img})
#     onnx_result =torch.from_numpy(onnx_result[0])
#
#     pred = non_max_suppression(onnx_result, 0.5, 0.45,  agnostic=False)
#
#     for i, det in enumerate(pred):
#         gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
#         if len(det):
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
#
#             for *xyxy, conf, cls in reversed(det):
#                 xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
#                 result.append([int(cls), xywh])
#
#     return result
