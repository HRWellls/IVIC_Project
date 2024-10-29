import cv2
import numpy as np
import torch

# 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

# 初始化视频捕捉
cap = cv2.VideoCapture("http://HRWellls:123456@10.162.151.142:8081/")  # 使用摄像头进行实时捕捉

# 初始化变量
prev_gray = None
prev_pts = None
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 首次初始化光流跟踪点
    if prev_gray is None:
        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        continue

    # 计算光流
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
    if next_pts is None or status is None:
        # 如果光流计算失败，跳过这一帧
        prev_gray = gray.copy()
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        continue
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    if good_new.shape[0] > 0 and good_old.shape[0] > 0:
        # 提取运动向量
        movement_vectors = good_new - good_old
        movement_magnitude = np.linalg.norm(movement_vectors, axis=1)
        # 计算平均速度
        average_speed = np.mean(movement_magnitude)

    # 物体检测
    results = model(frame)
    detections = results.xyxy[0]  # 获取检测结果

    # 绘制检测框和速度信息
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 显示速度信息
        if frame_count % 5 == 0:
            speed_text = f'Speed: {average_speed:.2f} px/frame'
            cv2.putText(frame, speed_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 更新上一帧
    prev_gray = gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

    # 显示结果
    cv2.imshow('Real-time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
