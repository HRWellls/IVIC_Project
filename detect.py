import cv2
import numpy as np
import torch

# 加载 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')

# 初始化视频捕捉
cap = cv2.VideoCapture(0)  # 使用摄像头进行实时捕捉

# 初始化变量
prev_gray = None
prev_pts = None

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
    good_new = next_pts[status == 1]
    good_old = prev_pts[status == 1]

    # 提取运动向量
    movement_vectors = good_new - good_old
    movement_magnitude = np.linalg.norm(movement_vectors, axis=1)

    # 物体检测
    results = model(frame)
    detections = results.xyxy[0]  # 获取检测结果

    # 绘制检测框
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # 更新上一帧
    prev_gray = gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

    # 显示结果
    cv2.imshow('Real-time Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
