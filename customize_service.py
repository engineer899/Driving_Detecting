import traceback
import os
import numpy as np
import time
import cv2
import onnxruntime as ort

# from input_reader import InputReader
# from models.experimental import attempt_load
# from tracker import Tracker

from utils1.general import check_img_size
from detect import detect_onnx


from model_service.pytorch_model_service import PTServingBaseService
# from utils1.torch_utils import TracedModel



class fatigue_driving_detection(PTServingBaseService):
    def __init__(self, model_name, model_path):
        # these three parameters are no need to modify
        self.model_name = model_name
        self.model_path = model_path
        self.capture = "test.mp4"

        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.first = True

        # 闭眼阈值
        self.th1 = 0.7
        # 打哈欠阈值
        self.th2 = 0.7
        # 打电话阈值
        self.th3 = 0.5
        # 左顾右盼阈值
        self.th4 = 0.8

        self.standard_pose = [180, 40, 80]
        self.normal_frame = 0
        self.look_around_frame = 0
        self.eyes_closed_frame = 0
        self.mouth_open_frame = 0
        self.use_phone_frame = 0
        # lStart, lEnd) = (42, 48)
        self.lStart = 42
        self.lEnd = 48
        # (rStart, rEnd) = (36, 42)
        self.rStart = 36
        self.rEnd = 42
        # (mStart, mEnd) = (49, 66)
        self.mStart = 49
        self.mEnd = 66
        self.EYE_AR_THRESH = 0.1
        self.MOUTH_AR_THRESH = 0.6
        self.frame_3s = self.fps * 3
        self.face_detect = 0

        self.imgsz = 640

        self.device = 'cpu'  # 大赛后台使用CPU判分

        # model = attempt_load(model_path, map_location=self.device)
        # self.stride = int(model.stride.max())
        # self.imgsz = check_img_size(self.imgsz, s=self.stride)
        #
        # self.model = TracedModel(model, self.device, self.imgsz)

        # self.model = TracedModel(model, self.device, self.imgsz)
        self.path = r"/home/mind/model/best_6M_980.onnx"
        cuda = False
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.model = ort.InferenceSession(self.path,providers=providers)#模型初始化


        self.stride = 32
        self.imgsz = check_img_size(self.imgsz, s=self.stride)


        self.need_reinit = 0
        self.failures = 0


    def _preprocess(self, data):
        # preprocessed_data = {}
        print("preprocessed", data)
        for k, v in data.items():
            for file_name, file_content in v.items():
                try:
                    try:
                        with open(self.capture, 'wb') as f:
                            file_content_bytes = file_content.read()
                            f.write(file_content_bytes)

                    except Exception:
                        return {"message": "There was an error loading the file"}

                    # self.capture = self.temp.name  # Pass temp.name to VideoCapture()
                except Exception:
                    return {"message": "There was an error processing the file"}
        return 'ok'


    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        # self.capture = path
        result = {"result": {"category": 0, "duration": 6000}}
        cap = cv2.VideoCapture(self.capture)


        # 在视频流的帧的宽度
        self.width = (int)(cap.get(3))
        # 在视频流的帧的高度
        self.height = (int)(cap.get(4))
        # 视频流的帧率
        self.fps = (int)(cap.get(5))
        self.frame_3s = self.fps * 3
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        now = time.time()

        total = int(self.total_frames / self.fps)

        all_frame_list = []

        for i in range(total):

            ff = int(i * self.fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, ff)
            ret, frame = cap.read()
            if ret:
                # 在这里对每一帧进行处理
                try:
                    if frame is not None:
                        # 剪裁主驾驶位
                        frame = frame[:, self.width - int(self.height * 1):self.width, :]
                        bbox = detect_onnx(self.model, frame, self.stride, self.imgsz)  # 目标检测结果获取
                        if len(bbox) == 0:
                            continue
                    for box in bbox:  # 统计检测结果
                        all_frame_list.append(box[0])
                        break
                    self.failures = 0

                except Exception as e:
                    if e.__class__ == KeyboardInterrupt:
                        print("Quitting")
                        break
            traceback.print_exc()
            self.failures += 1
            if self.failures > 30:  # 失败超过30次就默认返回
                break


        left_index = 0
        start_index = 0
        end_index = -100
        all_frame_list.append(-1)
        for i in range(len(all_frame_list)):
            if all_frame_list[i] != all_frame_list[left_index]:
                if all_frame_list[left_index] != 0 and i-left_index-1 > end_index-start_index:
                    start_index = left_index
                    end_index = i-1
                left_index = i

        start_frame_index = int(start_index * self.fps)
        end_frame_index = int(end_index * self.fps)

        count=1+end_index-start_index
        th=3
        if count >= th+1:  # 出现次数最多的状态，设置阈值
            category = all_frame_list[start_index]
        elif count == th:
            if all_frame_list[start_index] ==1:
                category = 1 if self.th1 == 0 else self.find(cap, start_frame_index, end_frame_index, self.fps,1, self.total_frames)
            elif all_frame_list[start_index] ==2:
                category = 2 if self.th2 == 0 else self.find(cap, start_frame_index, end_frame_index, self.fps, 2,self.total_frames)
            elif all_frame_list[start_index] ==3:
                category = 3 if self.th3 == 0 else self.find(cap, start_frame_index, end_frame_index, self.fps, 3,self.total_frames)
            else:
                category = 4 if self.th4 == 0 else self.find(cap,start_frame_index, end_frame_index, self.fps,4, self.total_frames)
        else:
            category = 0
        result['result']['category'] = str(category)

        cap.release()
        final_time = time.time()
        duration = int(np.round((final_time - now) * 1000))
        result['result']['duration'] = duration
        return result


    def _postprocess(self, data):
        os.remove(self.capture)
        return data

    def is_cls(self, cap, cls, index_):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index_)
        ret, frame = cap.read()
        # 在视频流的帧的宽度
        width = (int)(cap.get(3))
        # 在视频流的帧的高度
        height = (int)(cap.get(4))
        if ret:
            # 在这里对每一帧进行处理
            try:
                if frame is not None:
                    # 剪裁主驾驶位
                    frame = frame[:, width - int(height * 1):width, :]
                    bbox = detect_onnx(self.model, frame, 32, self.imgsz)  # 目标检测结果获取
                    if len(bbox) == 0:
                        return False
                    if bbox[0][0] == cls:
                        return True
                    else:
                        return False
                else:
                    return False
            except Exception as e:
                return False

    def find(self, cap, start, end, fps, cls, total):
        # print("find", fps)
        low = start - fps if start >= fps else 0
        high = start
        while low <= high:
            mid = (low + high) // 2  # 计算中间元素的索引
            if self.is_cls(cap, cls, mid):  # 找到目标值
                high = mid - 1
            else:  # 目标值在左侧子数组中
                low = mid + 1
        tot_f = start - high
        # print("前面", tot_f)

        low = end
        high = end + fps if start >= fps else total
        while low <= high:
            mid = (low + high) // 2  # 计算中间元素的索引
            if self.is_cls(cap, cls, mid):  # 找到目标值
                low = mid + 1
            else:  # 目标值在左侧子数组中
                high = mid - 1
        # print("后面", (high - end))

        tot_f = tot_f + (high - end)+1
        threshold = 0
        if cls == 1:
            threshold = self.th1
        elif cls == 2:
            threshold = self. th2
        elif cls == 3:
            threshold = self. th3
        else:
            threshold = self.th4

        if tot_f >= int(fps*threshold):
            # print("res", tot_f)
            return cls
        else:
            return 0
