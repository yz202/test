import cv2
import numpy as np

# 设置检测对象的阈值
thres = 0.45
# 设置非极大值抑制的阈值
nms_threshold = 0.2

# 初始化视频捕获对象，参数0表示使用计算机的默认摄像头
cap = cv2.VideoCapture(0)

# 加载COCO数据集的类别名称
classNames = []
classFile = 'data.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# 加载模型的配置文件和权重文件
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# 加载并配置检测模型
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)  # 设置输入图像的大小
net.setInputScale(1.0 / 127.5)  # 设置输入图像的缩放比例
net.setInputMean((127.5, 127.5, 127.5))  # 设置输入图像的均值
net.setInputSwapRB(True)  # 对于BGR图像，需要交换红色和蓝色通道

# 循环读取视频帧
while True:
    success, img = cap.read()  # 读取一帧图像
    if not success:
        break  # 如果读取失败，则退出循环

    # 使用模型进行对象检测
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    # 将bbox和confs转换为列表，并确保confs是浮点数列表
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    # 应用非极大值抑制
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    # 遍历检测到的对象，并绘制矩形框和类别文本
    for i in indices:
        i = i[0]  # NMSBoxes返回的索引是二维的，但这里我们只取第一个元素
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)  # 注意这里应该是y+h
        cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (x + 10, y + 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # 显示结果图像
    cv2.imshow("Output", img)

    # 按'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 释放视频捕获对象和销毁所有OpenCV窗口
cap.release()
cv2.destroyAllWindows()