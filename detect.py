import cv2
import numpy as np
import torch
import pyttsx3
import time
import threading

# 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

# 初始化视频捕捉
cap = cv2.VideoCapture("http://HRWellls:123456@10.162.151.142:8081/")  # 使用摄像头进行实时捕捉

# 初始化文本到语音引擎
engine = pyttsx3.init()

prev_gray = None
prev_pts = None

# 语音提示的时间控制
last_speech_time = time.time()
speech_interval = 10  # 语音提示间隔（秒）

# 定义方向提示函数
def get_direction(angle):
    if angle < -np.pi / 4 and angle >= -3 * np.pi / 4:
        return "上"
    elif angle >= -np.pi / 4 and angle < np.pi / 4:
        return "右"
    elif angle >= np.pi / 4 and angle < 3 * np.pi / 4:
        return "下"
    else:
        return "左"
    
def speak(direction_text, speed):
    engine.say(f'物体移动方向: {direction_text}，速度: {speed:.2f} 像素每帧')
    engine.runAndWait()

# 设定时间间隔（秒）
time_interval = 1.0  # 可以调整此值
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        continue

    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
    if next_pts is None or status is None:
        prev_gray = gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        continue

    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    # 物体检测
    results = model(frame)
    detections = results.xyxy[0]

    for i, (*box, conf, cls) in enumerate(detections):
        x1, y1, x2, y2 = map(int, box)
        center_new = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        if i < len(good_old):
            center_old = good_old[i]
            direction_vector = center_new - center_old
            direction_magnitude = np.linalg.norm(direction_vector)

            # 计算速度（假设时间间隔为1秒）
            speed = direction_magnitude / time_interval  # 速度计算
            speed_text = f'speed: {speed:.2f} px/s'  # 将速度显示为像素/秒

            if direction_magnitude > 1e-2:
                direction_unit = direction_vector / direction_magnitude
                angle = np.arctan2(direction_unit[1], direction_unit[0])
                direction_text = get_direction(angle)  # 获取大致方向
                cv2.putText(frame, speed_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # 语音输出
                current_time = time.time()
                if current_time - last_speech_time > speech_interval:
                    # 创建线程进行语音输出
                    threading.Thread(target=speak, args=(direction_text, speed)).start()
                    last_speech_time = current_time

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    prev_gray = gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

    cv2.imshow('Real-time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 键退出
        break

cap.release()
cv2.destroyAllWindows()
