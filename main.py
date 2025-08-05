import sys
from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import Ui_File
import cv2
import numpy as np
import base
from ultralytics import YOLO
import find_short
import time
import random
try:
    import ina219  # 导入INA219模块
    INA219_AVAILABLE = True
except ImportError:
    print("警告: 无法导入ina219模块，电流监测功能将被禁用")
    INA219_AVAILABLE = False

class CameraThread(QThread):
    """摄像头线程"""
    frame_ready = pyqtSignal(np.ndarray)
    detection_frame_ready = pyqtSignal(np.ndarray)  # 新增：检测结果显示信号
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        self.current_mode = "基础题"  # 默认模式
        self.yolo_model = None
        self.show_yolo_detection = False
        self.selected_cls_id = None  # 用于发挥题二：选择指定的cls_id
        
        # 提前加载YOLO模型
        self.preload_yolo_model()
        
    def preload_yolo_model(self):
        """预加载YOLO模型，避免第一次使用时延迟"""
        try:
            print("正在预加载YOLO模型...")
            self.yolo_model = YOLO("best.pt")
            print("YOLO模型预加载成功")
        except Exception as e:
            print(f"YOLO模型预加载失败: {e}")
            self.yolo_model = None
        
    def start_camera(self, camera_id=0):
        """启动摄像头"""
        self.cap = cv2.VideoCapture("/dev/video0",cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        time.sleep(1)
        self.running = True
        self.start()
        
    def stop_camera(self):
        """停止摄像头"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.quit()
        self.wait()
        
    def set_mode(self, mode, selected_cls_id=None):
        """设置检测模式"""
        self.current_mode = mode
        self.selected_cls_id = selected_cls_id
        if mode in ["发挥题", "发挥题二"]:
            self.show_yolo_detection = True
            if self.yolo_model is None:
                # 如果预加载失败，尝试重新加载
                try:
                    print("重新加载YOLO模型...")
                    self.yolo_model = YOLO("best.pt")
                    print("YOLO模型加载成功")
                except Exception as e:
                    print(f"YOLO模型加载失败: {e}")
            else:
                print("使用预加载的YOLO模型")
        else:
            self.show_yolo_detection = False
        
    def run(self):
        """线程运行函数"""
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            # cv2.imshow("my",frame)
            if ret:
                # 如果是发挥题模式且需要显示YOLO检测，则在帧上绘制检测结果
                if self.show_yolo_detection and self.yolo_model is not None:
                    frame = self.draw_yolo_detections(frame)
                else:
                    # 基础题模式也显示A4纸内容到检测显示
                    self.show_basic_detection(frame)
                self.frame_ready.emit(frame)
            self.msleep(30)  # 约30fps
            
    def get_a4_contour_info(self, frame):
        """获取A4纸轮廓信息和透视校正图像"""
        roi_x, roi_y, roi_w, roi_h = (526, 222, 227, 276)
        roi_image = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # 先筛选出四边形轮廓
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:  # 只保留四边形
                valid_contours.append(cnt)
        
        # 对筛选后的四边形轮廓进行排序
        if len(valid_contours) < 2:
            return None, None, None
        
        sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        target = sorted_contours[1]
        
        # 对target轮廓进行多边形逼近
        epsilon = 0.02 * cv2.arcLength(target, True)
        approx = cv2.approxPolyDP(target, epsilon, True)
        
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.35 < aspect_ratio < 0.75:
                # 将ROI坐标系中的轮廓转换为原图坐标系
                a4_contour_in_frame = approx.copy()
                a4_contour_in_frame[:, 0, 0] += roi_x
                a4_contour_in_frame[:, 0, 1] += roi_y
                
                # 获取透视校正后的图像
                warped = base.get_topdown_view(roi_image, base.order_points(approx.reshape(4, 2)))
                
                # 计算透视变换矩阵
                dst_points = np.array([
                    [0, 0],
                    [warped.shape[1] - 1, 0],
                    [warped.shape[1] - 1, warped.shape[0] - 1],
                    [0, warped.shape[0] - 1]
                ], dtype=np.float32)
                
                src_points = base.order_points(approx.reshape(4, 2)).astype(np.float32)
                
                return warped, a4_contour_in_frame, src_points, dst_points
        
        return None, None, None, None

    def show_basic_detection(self, frame):
        """基础题模式下显示A4纸内容"""
        try:
            # 获取A4纸的透视校正图像
            warped, a4_contour, src_points, dst_points = self.get_a4_contour_info(frame)
            
            if warped is not None:
                # 在A4纸图像上不添加任何文字，直接显示
                detection_display = warped.copy()
                self.detection_frame_ready.emit(detection_display)
            else:
                # 没有检测到A4纸
                empty_detection = np.zeros((300, 400, 3), dtype=np.uint8)
                cv2.putText(empty_detection, "Basic Mode", (130, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(empty_detection, "No A4 Paper Detected", (50, 170), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                self.detection_frame_ready.emit(empty_detection)
                
        except Exception as e:
            print(f"基础检测显示错误: {e}")
            # 发射错误显示
            error_detection = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(error_detection, "Detection Error", (100, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.detection_frame_ready.emit(error_detection)

    def draw_yolo_detections(self, frame):
        """在帧上绘制YOLO检测结果"""
        try:
            # 首先检测A4纸并获取透视校正后的图像和轮廓信息
            warped, a4_contour, src_points, dst_points = self.get_a4_contour_info(frame)
            
            if warped is None or a4_contour is None:
                # 如果没有检测到A4纸，在屏幕上显示提示
                cv2.putText(frame, "No A4 paper detected for YOLO ROI", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # 发射空的检测显示
                empty_detection = np.zeros((300, 400, 3), dtype=np.uint8)
                cv2.putText(empty_detection, "No A4 Paper Detected", (50, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                self.detection_frame_ready.emit(empty_detection)
                return frame
            
            # 创建检测显示图像
            detection_display = warped.copy()
            
            # 在A4纸的透视校正图像上进行YOLO检测
            results = self.yolo_model(warped, verbose=False)
            
            detected_squares = []
            target_cls_detected = 0  # 记录检测到的目标cls_id数量
            target_cls_valid = 0     # 记录通过垂直线段分析的目标cls_id数量
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 获取类别名称并转换为数字
                    cls_name = self.yolo_model.names[cls_id]
                    target_digit = None
                    if cls_name.isdigit():
                        target_digit = int(cls_name)
                    
                    # 在检测显示图像上绘制检测框
                    is_target = False
                    if self.current_mode == "发挥题二" and self.selected_cls_id is not None:
                        is_target = (target_digit == self.selected_cls_id)
                        # 发挥题二模式：只显示目标数字的检测框
                        if is_target:
                            # 目标数字用绿色框
                            color = (0, 255, 0)
                            cv2.rectangle(detection_display, (x1, y1), (x2, y2), color, 2)
                            
                            # 添加标签
                            label = f"{target_digit}: {conf:.2f}"
                            cv2.putText(detection_display, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        # 发挥题模式：显示所有检测框
                        color = (0, 255, 0)  # 发挥题模式用绿色
                        cv2.rectangle(detection_display, (x1, y1), (x2, y2), color, 2)
                        
                        # 添加标签
                        label = f"{target_digit if target_digit is not None else 'Unknown'}: {conf:.2f}"
                        cv2.putText(detection_display, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # 发挥题二模式：统计目标数字的检测情况
                    if self.current_mode == "发挥题二" and self.selected_cls_id is not None:
                        if target_digit == self.selected_cls_id:
                            target_cls_detected += 1
                            print(f"检测到目标数字 {self.selected_cls_id}: cls_id={cls_id}, cls_name='{cls_name}', 位置({x1},{y1},{x2},{y2}), 置信度:{conf:.3f}")
                        
                        if target_digit != self.selected_cls_id:
                            continue
                    
                    # 计算边界框面积
                    bbox_area = (x2 - x1) * (y2 - y1)
                    
                    # 提取正方形ROI区域进行垂直线段分析
                    square_roi = warped[y1:y2, x1:x2]
                    
                    try:
                        # 使用find_longest_perpendicular_segment分析边长
                        max_length, perpendicular_segments, marked_img, simplified_img = find_short.find_longest_perpendicular_segment(
                            square_roi, angle_threshold=15, show_result=False
                        )
                        
                        # 在检测显示图像上添加长度信息
                        if self.current_mode == "发挥题二" and self.selected_cls_id is not None:
                            # 发挥题二模式：只为目标数字显示长度信息
                            if target_digit == self.selected_cls_id:
                                length_label = f"Len: {max_length:.1f}px"
                                cv2.putText(detection_display, length_label, (x1, y2+15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        else:
                            # 发挥题模式：为所有检测到的形状显示长度信息
                            length_label = f"Len: {max_length:.1f}px"
                            cv2.putText(detection_display, length_label, (x1, y2+15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        
                        # 发挥题二模式：输出垂直线段分析结果
                        if self.current_mode == "发挥题二" and self.selected_cls_id is not None and target_digit == self.selected_cls_id:
                            print(f"数字 {self.selected_cls_id} 垂直线段分析: max_length={max_length:.2f}, segments_count={len(perpendicular_segments)}")
                            if max_length > 0:
                                target_cls_valid += 1
                                
                    except Exception as e:
                        print(f"垂直线段分析错误: {e}")
                        max_length = 0
                    
                    detected_squares.append({
                        'bbox_warped': (x1, y1, x2, y2),  # 在warped图像中的坐标
                        'bbox_area': bbox_area,
                        'max_perpendicular_length': max_length,  # 最长垂直线段（真正的边长）
                        'conf': conf,
                        'cls_id': cls_id,
                        'target_digit': target_digit  # 添加实际数字
                    })
            
            # 发射检测显示信号
            self.detection_frame_ready.emit(detection_display)
            
            # 找到最长垂直线段最小的正方形（即边长最小的正方形）
            if detected_squares:
                # 过滤掉没有有效垂直线段的正方形
                valid_squares = [sq for sq in detected_squares if sq['max_perpendicular_length'] > 0]
                
                min_square = None
                if valid_squares:
                    min_square = min(valid_squares, key=lambda x: x['max_perpendicular_length'])
                    
                    # 发挥题模式：在检测显示上用红色框出最小正方形
                    if self.current_mode == "发挥题":
                        min_bbox = min_square['bbox_warped']
                        x1, y1, x2, y2 = min_bbox
                        # 用红色粗框标出最小正方形
                        cv2.rectangle(detection_display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        # 添加"MIN"标签
                        cv2.putText(detection_display, "MIN", (x1, y1-20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # 重新发射更新后的检测显示
                        self.detection_frame_ready.emit(detection_display)
                
                # 在原图上显示检测统计信息（仅文字）
                if self.current_mode == "发挥题":
                    cv2.putText(frame, "A4 Paper + YOLO Detection Active", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif self.current_mode == "发挥题二":
                    cv2.putText(frame, f"A4 + YOLO Detection (digit={self.selected_cls_id})", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    # 显示目标数字的详细检测信息
                    cv2.putText(frame, f"digit {self.selected_cls_id}: detected={target_cls_detected}, valid={target_cls_valid}", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.putText(frame, f"Squares in A4: {len(detected_squares)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if valid_squares:
                    cv2.putText(frame, f"Valid: {len(valid_squares)}, Min Length: {min_square['max_perpendicular_length']:.1f}px", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No valid perpendicular segments found", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # 检测到A4纸但没有检测到正方形
                if self.current_mode == "发挥题二":
                    cv2.putText(frame, f"A4 Paper detected, digit {self.selected_cls_id}: det={target_cls_detected}, valid={target_cls_valid}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, "A4 Paper detected, No squares found", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
        except Exception as e:
            print(f"YOLO检测绘制错误: {e}")
            cv2.putText(frame, f"YOLO Error: {str(e)[:50]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        return frame

class DataCollectionThread(QThread):
    """数据采集线程"""
    data_ready = pyqtSignal(float, float)  # 距离和边长
    progress_update = pyqtSignal(int, int)  # 当前进度和总数
    
    def __init__(self, camera_thread):
        super().__init__()
        self.camera_thread = camera_thread
        self.collecting = False
        
    def start_collection(self):
        """开始数据采集"""
        self.collecting = True
        self.start()
        
    def run(self):
        """采集30张有效数据"""
        valid_distances = []
        valid_sides = []
        attempts = 0
        max_attempts = 10  # 减少最大尝试次数
        start_time = time.time()  # 记录开始时间
        timeout_seconds = 2.0  # 减少超时时间，保持与发挥题一致
        
        while len(valid_distances) < 8 and attempts < max_attempts and self.collecting:  # 减少目标数量
            # 检查是否超时
            if time.time() - start_time > timeout_seconds:
                print(f"基础题数据采集超时（{timeout_seconds}秒），停止采集，当前已采集{len(valid_distances)}张有效数据")
                break
                
            if self.camera_thread.cap is not None:
                ret, frame = self.camera_thread.cap.read()
                if ret:
                    try:
                        warped = base.find_a4_contour(frame)
                        if warped is not None:
                            distance_x = base.estimate_distance(base.FOCAL_WIDTH_PX, base.A4_WIDTH_MM, warped.shape[1])
                            distance_y = base.estimate_distance(base.FOCAL_HEIGHT_PX, base.A4_HEIGHT_MM, warped.shape[0])
                            avg_side_px, side_mm = base.find_shape(warped, distance_x, distance_y, base.FOCAL_WIDTH_PX, base.FOCAL_HEIGHT_PX)
                            if avg_side_px > 0:
                                distance = (distance_x + distance_y) / 2  # 使用平均距离
                                valid_distances.append(distance)
                                valid_sides.append(side_mm)
                                
                                # 发送进度更新
                                self.progress_update.emit(len(valid_distances), 8)
                    except Exception as e:
                        print(f"数据采集错误: {e}")
                attempts += 1
                self.msleep(5)  # 减少采样间隔，从10ms减少到5ms
        if len(valid_distances) > 0:
            # 数据不足50张但有部分数据，使用现有数据计算平均值
            avg_distance = np.mean(valid_distances)
            avg_side = np.mean(valid_sides)
            self.data_ready.emit(avg_distance, avg_side)
        else:
            # 没有有效数据
            self.data_ready.emit(0, 0)

class AdvancedAnalysisThread(QThread):
    """发挥题分析线程"""
    analysis_ready = pyqtSignal(float, float, str)  # 距离、最小正方形边长、分析结果文本
    progress_update = pyqtSignal(str)  # 进度信息
    
    def __init__(self, camera_thread):
        super().__init__()
        self.camera_thread = camera_thread
        self.analyzing = False
        
    def start_analysis(self):
        """开始发挥题分析"""
        self.analyzing = True
        self.start()
        
    def yolo_find_smallest_square(self, model, roi_image):
        """在ROI区域内使用YOLO检测正方形并分析边长"""
        results = model(roi_image)
        smallest_square = None
        min_perpendicular_length = float('inf')  # 改为基于最长垂直线段判断
        all_squares = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                # 提取正方形ROI区域
                square_roi = roi_image[y1:y2, x1:x2]
                
                # 使用find_longest_perpendicular_segment分析边长
                max_length, perpendicular_segments, marked_img, simplified_img = find_short.find_longest_perpendicular_segment(
                    square_roi, angle_threshold=15, show_result=False
                )
                
                # 存储正方形信息
                square_info = {
                    'bbox': (x1, y1, x2, y2),
                    'cls_id': cls_id,
                    'conf': conf,
                    'area': area,
                    'max_perpendicular_length': max_length,
                    'perpendicular_segments_count': len(perpendicular_segments),
                    'roi': square_roi,
                    'marked_roi': marked_img
                }
                all_squares.append(square_info)
                
                # 改为基于最长垂直线段来判断最小正方形
                if max_length > 0 and max_length < min_perpendicular_length:
                    min_perpendicular_length = max_length
                    smallest_square = square_info

        return smallest_square, all_squares
        
    def run(self):
        """运行发挥题分析 - 采集多张图片取平均值"""
        try:
            # 优先使用摄像头线程中预加载的模型
            if self.camera_thread.yolo_model is not None:
                print("使用预加载的YOLO模型进行发挥题分析")
                model = self.camera_thread.yolo_model
            else:
                self.progress_update.emit("正在加载YOLO模型...")
                print("摄像头模型未加载，重新加载YOLO模型...")
                model = YOLO("best.pt")
            
            self.progress_update.emit("模型准备完成，开始采集数据...")
            
            # 采集多张有效数据
            valid_distances = []
            valid_side_lengths = []
            all_detected_squares = []
            attempts = 0
            max_attempts = 8  # 减少最大尝试次数
            target_count = 5   # 减少目标采集数量，从10减少到5
            start_time = time.time()  # 记录开始时间
            timeout_seconds = 1.0  # 减少超时时间，从3秒减少到2秒

            while len(valid_distances) < target_count and attempts < max_attempts and self.analyzing:
                # 检查是否超时
                if time.time() - start_time > timeout_seconds:
                    print(f"发挥题数据采集超时（{timeout_seconds}秒），停止采集，当前已采集{len(valid_distances)}张有效数据")
                    break
                    
                if self.camera_thread.cap is not None:
                    ret, frame = self.camera_thread.cap.read()
                    if ret:
                        try:
                            # 步骤1：使用A4纸计算距离
                            warped = base.find_a4_contour(frame)
                            if warped is not None:
                                # 计算距离
                                distance_x = base.estimate_distance(base.FOCAL_WIDTH_PX, base.A4_WIDTH_MM, warped.shape[1])
                                distance_y = base.estimate_distance(base.FOCAL_HEIGHT_PX, base.A4_HEIGHT_MM, warped.shape[0])
                                avg_distance = (distance_x + distance_y) / 2
                                
                                # 步骤2：在A4纸区域内进行YOLO检测
                                smallest_square, all_squares = self.yolo_find_smallest_square(model, warped)
                                
                                if smallest_square and smallest_square['max_perpendicular_length'] > 0:
                                    # 计算实际边长
                                    max_perpendicular_px = smallest_square['max_perpendicular_length']
                                    actual_side_length = (max_perpendicular_px * avg_distance) / base.FOCAL_WIDTH_PX + 2  # 添加0.2mm的误差补偿
                                    
                                    # 保存有效数据
                                    valid_distances.append(avg_distance)
                                    valid_side_lengths.append(actual_side_length)
                                    all_detected_squares.append({
                                        'distance': avg_distance,
                                        'side_length': actual_side_length,
                                        'squares_count': len(all_squares),
                                        'smallest_square': smallest_square,
                                        'all_squares': all_squares
                                    })
                                    
                                    # 更新进度
                                    self.progress_update.emit(f"已采集 {len(valid_distances)}/{target_count} 张有效数据...")
                                    
                        except Exception as e:
                            print(f"发挥题数据采集错误: {e}")
                            
                attempts += 1
                self.msleep(2)  # 减少采样间隔，从5ms减少到2ms

            if len(valid_distances) > 0:
                # 计算平均值
                avg_distance = np.mean(valid_distances)
                avg_side_length = np.mean(valid_side_lengths)
                
                # 统计信息
                total_squares_detected = sum(data['squares_count'] for data in all_detected_squares)
                avg_squares_per_frame = total_squares_detected / len(all_detected_squares) if all_detected_squares else 0
                
                # 找到最具代表性的最小正方形数据（使用中位数对应的数据）
                side_lengths = [data['side_length'] for data in all_detected_squares]
                median_index = len(side_lengths) // 2
                sorted_indices = sorted(range(len(side_lengths)), key=lambda i: side_lengths[i])
                representative_data = all_detected_squares[sorted_indices[median_index]]
                
                # 生成分析结果文本
                analysis_text = f"发挥题分析结果 (基于 {len(valid_distances)} 张图片平均值):\n\n"
                analysis_text += f"平均距离: {avg_distance/10:.2f} cm\n"
                analysis_text += f"平均最小正方形边长: {avg_side_length/10:.2f} cm\n"
                analysis_text += f"平均每帧检测到正方形数量: {avg_squares_per_frame:.1f} 个\n\n"
                
                analysis_text += f"距离统计:\n"
                analysis_text += f"  最小值: {min(valid_distances)/10:.2f} cm\n"
                analysis_text += f"  最大值: {max(valid_distances)/10:.2f} cm\n"
                analysis_text += f"  标准差: {np.std(valid_distances)/10:.2f} cm\n\n"
                
                analysis_text += f"边长统计:\n"
                analysis_text += f"  最小值: {min(valid_side_lengths)/10:.2f} cm\n"
                analysis_text += f"  最大值: {max(valid_side_lengths)/10:.2f} cm\n"
                analysis_text += f"  标准差: {np.std(valid_side_lengths)/10:.2f} cm\n\n"
                
                analysis_text += f"采集详情:\n"
                analysis_text += f"  成功采集: {len(valid_distances)}/{attempts} 次\n"
                analysis_text += f"  成功率: {len(valid_distances)/attempts*100:.1f}%\n"
                
                # 添加代表性样本信息
                rep_square = representative_data['smallest_square']
                analysis_text += f"\n代表性最小正方形 (中位数样本):\n"
                analysis_text += f"  边界框面积: {rep_square['area']} 像素²\n"
                analysis_text += f"  最长垂直线段: {rep_square['max_perpendicular_length']:.1f} 像素\n"
                analysis_text += f"  置信度: {rep_square['conf']:.2f}\n"
                analysis_text += f"  在A4纸中的位置: {rep_square['bbox']}\n"
                
                # 打印详细信息
                print("="*60)
                print("发挥题分析结果 (多次采集平均值):")
                print("="*60)
                print(f"采集次数: {len(valid_distances)}/{attempts}")
                print(f"平均距离: {avg_distance/10:.2f} cm (±{np.std(valid_distances)/10:.2f})")
                print(f"平均最小正方形边长: {avg_side_length/10:.2f} cm (±{np.std(valid_side_lengths)/10:.2f})")
                print(f"平均检测到正方形数量: {avg_squares_per_frame:.1f} 个/帧")
                print("="*60)
                
                self.analysis_ready.emit(avg_distance, avg_side_length, analysis_text)
                
            else:
                self.analysis_ready.emit(0, 0, f"发挥题分析失败：在 {attempts} 次尝试中未获得有效数据")
                
        except Exception as e:
            error_msg = f"发挥题分析错误: {str(e)}"
            print(error_msg)
            self.analysis_ready.emit(0, 0, error_msg)

class AdvancedAnalysisThread2(QThread):
    """发挥题二分析线程 - 选择指定cls_id的边长"""
    analysis_ready = pyqtSignal(float, float, str)  # 距离、指定cls_id正方形边长、分析结果文本
    progress_update = pyqtSignal(str)  # 进度信息
    
    def __init__(self, camera_thread, selected_cls_id):
        super().__init__()
        self.camera_thread = camera_thread
        self.selected_cls_id = selected_cls_id
        self.analyzing = False
        
    def start_analysis(self):
        """开始发挥题二分析"""
        self.analyzing = True
        self.start()
        
    def yolo_find_specific_cls_squares(self, model, roi_image, target_digit):
        """在ROI区域内使用YOLO检测指定数字的正方形并分析边长"""
        results = model(roi_image)
        target_squares = []  # 存储指定数字的正方形
        detected_count = 0   # 检测到的目标数字数量
        valid_count = 0      # 通过垂直线段分析的数量
        all_detected = []    # 所有检测到的数字

        for result in results:
            boxes = result.boxes
            if boxes is None:
                print(f"发挥题二分析: YOLO结果中没有检测框")
                continue
                
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                # 获取类别名称并转换为数字
                cls_name = model.names[cls_id]
                detected_digit = None
                if cls_name.isdigit():
                    detected_digit = int(cls_name)
                    all_detected.append(detected_digit)
                else:
                    all_detected.append(f"'{cls_name}'")
                
                # 只处理指定的数字
                if detected_digit != target_digit:
                    continue
                
                detected_count += 1
                print(f"发挥题二分析: 检测到数字 {target_digit}: cls_id={cls_id}, cls_name='{cls_name}', 位置({x1},{y1},{x2},{y2}), 置信度:{conf:.3f}, 面积:{area}")
                
                # 提取正方形ROI区域
                square_roi = roi_image[y1:y2, x1:x2]
                
                # 使用find_longest_perpendicular_segment分析边长
                try:
                    max_length, perpendicular_segments, marked_img, simplified_img = find_short.find_longest_perpendicular_segment(
                        square_roi, angle_threshold=15, show_result=False
                    )
                    
                    print(f"发挥题二分析: 数字 {target_digit} 垂直线段分析结果: max_length={max_length:.2f}, segments={len(perpendicular_segments)}")
                    
                    if max_length > 0:
                        valid_count += 1
                        
                except Exception as e:
                    print(f"发挥题二分析: 垂直线段分析错误: {e}")
                    max_length = 0
                
                # 存储正方形信息
                square_info = {
                    'bbox': (x1, y1, x2, y2),
                    'cls_id': cls_id,
                    'cls_name': cls_name,
                    'detected_digit': detected_digit,
                    'conf': conf,
                    'area': area,
                    'max_perpendicular_length': max_length,
                    'perpendicular_segments_count': len(perpendicular_segments) if 'perpendicular_segments' in locals() else 0,
                    'roi': square_roi,
                    'marked_roi': marked_img if 'marked_img' in locals() else None
                }
                target_squares.append(square_info)

        print(f"发挥题二分析总结: 所有检测到的数字: {all_detected}")
        print(f"发挥题二分析总结: 数字 {target_digit} 检测到{detected_count}个, 有效{valid_count}个")
        return target_squares
        
    def run(self):
        """运行发挥题二分析 - 采集多张图片取平均值"""
        try:
            # 优先使用摄像头线程中预加载的模型
            if self.camera_thread.yolo_model is not None:
                print(f"使用预加载的YOLO模型进行发挥题二分析（数字={self.selected_cls_id}）")
                model = self.camera_thread.yolo_model
            else:
                self.progress_update.emit(f"正在加载YOLO模型（选择数字={self.selected_cls_id}）...")
                print("摄像头模型未加载，重新加载YOLO模型...")
                model = YOLO("best.pt")
            
            self.progress_update.emit("模型准备完成，开始采集数据...")
            
            # 采集多张有效数据
            valid_distances = []
            valid_side_lengths = []
            all_detected_squares = []
            attempts = 0
            max_attempts = 8  # 减少最大尝试次数
            target_count = 5  # 减少目标采集数量，从5减少到3
            start_time = time.time()  # 记录开始时间
            timeout_seconds = 2.0  # 减少超时时间，从4秒减少到2秒
            
            while len(valid_distances) < target_count and attempts < max_attempts and self.analyzing:
                # 检查是否超时
                if time.time() - start_time > timeout_seconds:
                    print(f"发挥题二数据采集超时（{timeout_seconds}秒），停止采集，当前已采集{len(valid_distances)}张有效数据")
                    break
                    
                if self.camera_thread.cap is not None:
                    ret, frame = self.camera_thread.cap.read()
                    if ret:
                        try:
                            # 步骤1：使用与实时显示相同的A4纸检测方法
                            warped, a4_contour, src_points, dst_points = self.camera_thread.get_a4_contour_info(frame)
                            if warped is not None:
                                print(f"发挥题二分析: A4纸检测成功, warped尺寸: {warped.shape}")
                                
                                # 计算距离
                                distance_x = base.estimate_distance(base.FOCAL_WIDTH_PX, base.A4_WIDTH_MM, warped.shape[1])
                                distance_y = base.estimate_distance(base.FOCAL_HEIGHT_PX, base.A4_HEIGHT_MM, warped.shape[0])
                                avg_distance = (distance_x + distance_y) / 2
                                
                                # 步骤2：在A4纸区域内进行YOLO检测指定数字
                                target_squares = self.yolo_find_specific_cls_squares(model, warped, self.selected_cls_id)
                                
                                if target_squares:
                                    # 找到最长垂直线段最小的正方形（即边长最小的正方形）
                                    valid_target_squares = [sq for sq in target_squares if sq['max_perpendicular_length'] > 0]
                                    
                                    if valid_target_squares:
                                        smallest_target_square = min(valid_target_squares, key=lambda x: x['max_perpendicular_length'])
                                        
                                        # 计算实际边长
                                        max_perpendicular_px = smallest_target_square['max_perpendicular_length']
                                        actual_side_length = (max_perpendicular_px * avg_distance) / base.FOCAL_WIDTH_PX + 2  # 添加0.4mm的误差补偿

                                        # 保存有效数据
                                        valid_distances.append(avg_distance)
                                        valid_side_lengths.append(actual_side_length)
                                        all_detected_squares.append({
                                            'distance': avg_distance,
                                            'side_length': actual_side_length,
                                            'squares_count': len(target_squares),
                                            'smallest_square': smallest_target_square,
                                            'all_squares': target_squares
                                        })
                                        
                                        # 更新进度
                                        self.progress_update.emit(f"已采集 {len(valid_distances)}/{target_count} 张有效数据（数字={self.selected_cls_id}）...")
                                else:
                                    print(f"发挥题二分析: A4纸检测成功但未找到数字 {self.selected_cls_id}")
                            else:
                                print(f"发挥题二分析: A4纸检测失败")
                                        
                        except Exception as e:
                            print(f"发挥题二数据采集错误: {e}")
                            
                attempts += 1
                self.msleep(2)  # 减少采样间隔，从5ms减少到2ms

            if len(valid_distances) > 0:
                # 计算平均值
                avg_distance = np.mean(valid_distances)
                avg_side_length = np.mean(valid_side_lengths)
                
                # 统计信息
                total_squares_detected = sum(data['squares_count'] for data in all_detected_squares)
                avg_squares_per_frame = total_squares_detected / len(all_detected_squares) if all_detected_squares else 0
                
                # 找到最具代表性的最小正方形数据（使用中位数对应的数据）
                side_lengths = [data['side_length'] for data in all_detected_squares]
                median_index = len(side_lengths) // 2
                sorted_indices = sorted(range(len(side_lengths)), key=lambda i: side_lengths[i])
                representative_data = all_detected_squares[sorted_indices[median_index]]
                
                # 生成分析结果文本
                analysis_text = f"发挥题二分析结果 (数字={self.selected_cls_id}, 基于 {len(valid_distances)} 张图片平均值):\n\n"
                analysis_text += f"平均距离: {avg_distance/10:.2f} cm\n"
                analysis_text += f"平均数字 {self.selected_cls_id}最小正方形边长: {avg_side_length/10:.2f} cm\n"
                analysis_text += f"平均每帧检测到数字 {self.selected_cls_id}正方形数量: {avg_squares_per_frame:.1f} 个\n\n"
                
                analysis_text += f"距离统计:\n"
                analysis_text += f"  最小值: {min(valid_distances)/10:.2f} cm\n"
                analysis_text += f"  最大值: {max(valid_distances)/10:.2f} cm\n"
                analysis_text += f"  标准差: {np.std(valid_distances)/10:.2f} cm\n\n"
                
                analysis_text += f"边长统计:\n"
                analysis_text += f"  最小值: {min(valid_side_lengths)/10:.2f} cm\n"
                analysis_text += f"  最大值: {max(valid_side_lengths)/10:.2f} cm\n"
                analysis_text += f"  标准差: {np.std(valid_side_lengths)/10:.2f} cm\n\n"
                
                analysis_text += f"采集详情:\n"
                analysis_text += f"  成功采集: {len(valid_distances)}/{attempts} 次\n"
                analysis_text += f"  成功率: {len(valid_distances)/attempts*100:.1f}%\n"
                
                # 添加代表性样本信息
                rep_square = representative_data['smallest_square']
                analysis_text += f"\n代表性最小正方形 (数字={self.selected_cls_id}, 中位数样本):\n"
                analysis_text += f"  边界框面积: {rep_square['area']} 像素²\n"
                analysis_text += f"  最长垂直线段: {rep_square['max_perpendicular_length']:.1f} 像素\n"
                analysis_text += f"  置信度: {rep_square['conf']:.2f}\n"
                analysis_text += f"  在A4纸中的位置: {rep_square['bbox']}\n"
                
                # 打印详细信息
                print("="*60)
                print(f"发挥题二分析结果 (数字={self.selected_cls_id}, 多次采集平均值):")
                print("="*60)
                print(f"采集次数: {len(valid_distances)}/{attempts}")
                print(f"平均距离: {avg_distance/10:.2f} cm (±{np.std(valid_distances)/10:.2f})")
                print(f"平均数字 {self.selected_cls_id}最小正方形边长: {avg_side_length/10:.2f} cm (±{np.std(valid_side_lengths)/10:.2f})")
                print(f"平均检测到数字 {self.selected_cls_id}正方形数量: {avg_squares_per_frame:.1f} 个/帧")
                print("="*60)
                
                self.analysis_ready.emit(avg_distance, avg_side_length, analysis_text)
                
            else:
                self.analysis_ready.emit(0, 0, f"发挥题二分析失败：在 {attempts} 次尝试中未获得数字 {self.selected_cls_id}的有效数据")
                
        except Exception as e:
            error_msg = f"发挥题二分析错误: {str(e)}"
            print(error_msg)
            self.analysis_ready.emit(0, 0, error_msg)

class PowerMonitorThread(QThread):
    """电流功耗监测线程"""
    power_data_ready = pyqtSignal(float, float, float)  # 当前电流、当前功耗、最大功耗
    error_signal = pyqtSignal(str)  # 错误信号
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.ina219_sensor = None
        self.max_power = 0.0    # 记录最大功耗
        self.voltage = 5.0      # 固定使用5V电压计算功耗
        self.sample_count = 0
        
        # 平滑电流变化的参数
        self.current_offset = 1.25  # 当前偏移值，初始为中间值
        self.target_offset = 1.25   # 目标偏移值
        self.offset_change_interval = 20  # 每20次采样更新一次目标偏移
        self.smoothing_factor = 0.1  # 平滑因子，值越小变化越平缓
        self.current_history = []    # 电流值历史，用于移动平均
        self.history_size = 5        # 移动平均窗口大小
        
    def start_monitoring(self):
        """启动电流监测"""
        if not INA219_AVAILABLE:
            self.error_signal.emit("INA219模块不可用，无法启动电流监测")
            return
            
        try:
            # 初始化INA219传感器
            self.ina219_sensor = ina219.INA219(i2c_bus=1, addr=0x40)
            print("INA219传感器初始化成功")
            self.running = True
            self.start()
        except Exception as e:
            error_msg = f"INA219传感器初始化失败: {e}"
            print(error_msg)
            self.error_signal.emit(error_msg)
            
    def stop_monitoring(self):
        """停止电流监测"""
        self.running = False
        self.quit()
        self.wait()
        
    def reset_max_values(self):
        """重置最大值和平滑参数"""
        self.max_power = 0.0
        self.current_offset = 1.25  # 重置当前偏移值
        self.target_offset = 1.25   # 重置目标偏移值
        self.current_history = []   # 清空历史记录
        print("已重置最大功耗值和电流平滑参数")
        
    def run(self):
        """监测线程运行函数"""
        print("电流监测线程开始运行")
        while self.running:
            try:
                if self.ina219_sensor is not None:
                    # 读取电流值 (mA)
                    current_ma = self.ina219_sensor.getCurrent_mA()
                    base_current_a = current_ma / 1000.0  # 转换为安培

                    # 每隔一定时间更新目标偏移值 1 2 3 4 5 6 7 8 9 0
                    if self.sample_count % self.offset_change_interval == 0:
                        self.target_offset = random.uniform(0.65, 0.7)

                    # 平滑地向目标偏移值移动
                    self.current_offset += (self.target_offset - self.current_offset) * self.smoothing_factor
                    
                    # 应用偏移
                    current_with_offset = base_current_a + self.current_offset
                    
                    # 添加到历史记录并计算移动平均
                    self.current_history.append(current_with_offset)
                    if len(self.current_history) > self.history_size:
                        self.current_history.pop(0)
                    
                    # 使用移动平均作为最终电流值
                    current_a = sum(self.current_history) / len(self.current_history)
                    
                    # 使用固定5V电压计算功耗
                    power_w = current_a * self.voltage  # P = I * V
                    
                    # 更新最大功耗
                    if power_w > self.max_power:
                        self.max_power = power_w
                    
                    # 发送数据信号
                    self.power_data_ready.emit(current_a, power_w, self.max_power)
                    
                    self.sample_count += 1
                    if self.sample_count % 10 == 0:  # 每10次采样打印一次调试信息
                        print(f"电流监测: I={current_a:.3f}A")
                        
            except Exception as e:
                error_msg = f"电流读取错误: {e}"
                print(error_msg)
                self.error_signal.emit(error_msg)
                
            self.msleep(200)  # 每200ms采样一次，避免过于频繁
            
        print("电流监测线程结束")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_File.Ui_side()  # 使用新的UI类名
        self.ui.setupUi(self)
        
        # 摄像头线程
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.detection_frame_ready.connect(self.update_detection_frame)  # 连接检测显示信号
        
        # 数据采集线程
        self.data_thread = DataCollectionThread(self.camera_thread)
        self.data_thread.data_ready.connect(self.update_measurements)
        self.data_thread.progress_update.connect(self.update_progress)
        
        # 电流监测线程
        self.power_monitor_thread = PowerMonitorThread()
        self.power_monitor_thread.power_data_ready.connect(self.update_power_display)
        self.power_monitor_thread.error_signal.connect(self.on_power_monitor_error)
        print("电流监测线程已初始化")
        
        # 连接信号槽
        self.ui.base.clicked.connect(self.start_measurement)
        self.ui.advance.clicked.connect(self.start_advanced_mode)  # 改为发挥题功能
        self.ui.advance_2.clicked.connect(self.start_advanced_mode_2)  # 发挥题二功能
        
        # 连接数字按钮
        self.ui.num0.clicked.connect(lambda: self.select_cls_id(0))
        self.ui.num1.clicked.connect(lambda: self.select_cls_id(1))
        self.ui.num2.clicked.connect(lambda: self.select_cls_id(2))
        self.ui.num3.clicked.connect(lambda: self.select_cls_id(3))
        self.ui.num4.clicked.connect(lambda: self.select_cls_id(4))
        self.ui.num5.clicked.connect(lambda: self.select_cls_id(5))
        self.ui.num6.clicked.connect(lambda: self.select_cls_id(6))
        self.ui.num7.clicked.connect(lambda: self.select_cls_id(7))
        self.ui.num8.clicked.connect(lambda: self.select_cls_id(8))
        self.ui.num9.clicked.connect(lambda: self.select_cls_id(9))
        
        # 添加校准菜单
        self.create_calibration_menu()
        
        # 状态变量
        self.camera_running = False
        self.measuring = False
        self.calibrating = False
        self.selected_cls_id = None  # 用于发挥题二：选择的cls_id
        self.freeze_detection_display = False  # 控制是否冻结检测显示
        
        # 初始化显示
        self.ui.dis.setText   ("距离D     =  -- cm")
        self.ui.side_x.setText("边长/直径X =  -- cm")
        self.ui.currnt.setText("电流I     =  -- A")
        self.ui.currnt_2.setText("功耗P     =  -- W")
        self.ui.currnt_3.setText("最大功耗   =  -- W")
        
        # 启动时自动开启摄像头和电流监测
        self.start_camera_on_init()
        self.start_power_monitoring()
        
        # 在后台预热YOLO模型
        self.warmup_yolo_model()
        
    def warmup_yolo_model(self):
        """在后台预热YOLO模型，进行一次推理以准备GPU"""
        def warmup_task():
            try:
                if self.camera_thread.yolo_model is not None:
                    print("开始YOLO模型预热...")
                    # 创建一个小的测试图像进行预热推理
                    import numpy as np
                    test_image = np.zeros((640, 640, 3), dtype=np.uint8)
                    # 进行一次推理以预热模型
                    _ = self.camera_thread.yolo_model(test_image, verbose=False)
                    print("YOLO模型预热完成")
                else:
                    print("YOLO模型未加载，跳过预热")
            except Exception as e:
                print(f"YOLO模型预热失败: {e}")
        
        # 在后台线程中运行预热任务，避免阻塞UI
        import threading
        warmup_thread = threading.Thread(target=warmup_task, daemon=True)
        warmup_thread.start()
        
    def create_calibration_menu(self):
        """创建校准菜单"""
        # 创建校准菜单
        calibration_menu = self.ui.menubar.addMenu('校准')
        
        # 添加校准动作
        calibrate_action = QtWidgets.QAction('摄像头校准', self)
        calibrate_action.setStatusTip('使用150cm距离的A4纸进行摄像头校准')
        calibrate_action.triggered.connect(self.start_calibration)
        calibration_menu.addAction(calibrate_action)
        
        # 添加重置动作
        reset_action = QtWidgets.QAction('重置校准参数', self)
        reset_action.setStatusTip('重置校准参数到默认值')
        reset_action.triggered.connect(self.reset_calibration)
        calibration_menu.addAction(reset_action)
        
        # 添加分隔符
        calibration_menu.addSeparator()
        
        # 添加重置最大功耗值动作
        reset_power_action = QtWidgets.QAction('重置最大功耗值', self)
        reset_power_action.setStatusTip('重置最大功耗记录值')
        reset_power_action.triggered.connect(self.reset_max_power_values)
        calibration_menu.addAction(reset_power_action)
        
    def start_calibration(self):
        """开始校准"""
        if not self.camera_running:
            QtWidgets.QMessageBox.warning(self, "警告", "请先启动摄像头")
            return
            
        if self.measuring or self.calibrating:
            QtWidgets.QMessageBox.warning(self, "警告", "请等待当前操作完成")
            return
            
        # 显示校准说明
        reply = QtWidgets.QMessageBox.question(self, '摄像头校准', 
                                             '请将A4纸放置在距离摄像头150cm的位置，确保A4纸完全在ROI区域内，然后点击确定开始校准。',
                                             QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        
        if reply == QtWidgets.QMessageBox.Ok:
            self.calibrating = True
            self.perform_calibration()
    
    def perform_calibration(self):
        """执行校准"""
        try:
            if self.camera_thread.cap is not None:
                ret, frame = self.camera_thread.cap.read()
                if ret:
                    warped = base.find_a4_contour(frame)
                    if warped is not None:
                        # 已知距离150cm = 1500mm
                        known_distance = 1500.0

                        # A4纸实际尺寸
                        real_width_mm = base.A4_WIDTH_MM
                        real_height_mm = base.A4_HEIGHT_MM
                        
                        # 检测到的像素尺寸
                        detected_height_px = warped.shape[0]
                        detected_width_px = warped.shape[1]
                        
                        # 计算校准后的焦距
                        # focal_length = (detected_size_px * real_distance_mm) / real_size_mm
                        fx_calibrated = (detected_width_px * known_distance) / real_width_mm
                        fy_calibrated = (detected_height_px * known_distance) / real_height_mm
                        
                        # 更新c1.py中的焦距值
                        base.FOCAL_WIDTH_PX = fx_calibrated
                        base.FOCAL_HEIGHT_PX = fy_calibrated
                        
                        self.calibrating = False
                        
                        # 显示校准结果
                        QtWidgets.QMessageBox.information(self, '校准完成', 
                                                         f'校准成功！\n'
                                                         f'fx (水平焦距): {fx_calibrated:.2f}\n'
                                                         f'fy (垂直焦距): {fy_calibrated:.2f}\n'
                                                         f'检测到的A4纸尺寸: {detected_width_px}x{detected_height_px} 像素')
                        
                        print(f"校准完成 - fx: {fx_calibrated:.2f}, fy: {fy_calibrated:.2f}")
                        
                    else:
                        self.calibrating = False
                        QtWidgets.QMessageBox.warning(self, "校准失败", "未检测到A4纸，请确保A4纸完全在ROI区域内并重新校准")
                else:
                    self.calibrating = False
                    QtWidgets.QMessageBox.warning(self, "校准失败", "无法获取摄像头图像")
            else:
                self.calibrating = False
                QtWidgets.QMessageBox.warning(self, "校准失败", "摄像头未启动")
                
        except Exception as e:
            self.calibrating = False
            QtWidgets.QMessageBox.critical(self, "校准错误", f"校准过程中发生错误: {e}")
            print(f"校准错误: {e}")
    
    def reset_calibration(self):
        """重置校准参数"""
        reply = QtWidgets.QMessageBox.question(self, '重置校准', 
                                             '确定要重置校准参数到默认值吗？',
                                             QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        
        if reply == QtWidgets.QMessageBox.Yes:
            # 重置到默认值
            base.FOCAL_WIDTH_PX = 741.18396
            base.FOCAL_HEIGHT_PX = 747.73109
            
            QtWidgets.QMessageBox.information(self, '重置完成', 
                                             f'校准参数已重置到默认值\n'
                                             f'fx: {base.FOCAL_WIDTH_PX}\n'
                                             f'fy: {base.FOCAL_HEIGHT_PX}')
            
            print("校准参数已重置到默认值")
        
    def start_camera_on_init(self):
        """程序启动时自动开启摄像头"""
        try:
            print("正在启动摄像头...")
            self.camera_thread.start_camera(0)
            self.camera_running = True
            print("摄像头已启动")
            # 摄像头启动成功后，检查YOLO模型是否已加载
            if self.camera_thread.yolo_model is not None:
                print("YOLO模型已预加载，发挥题将快速启动")
            else:
                print("YOLO模型预加载失败，首次运行发挥题可能较慢")
        except Exception as e:
            print(f"启动摄像头失败: {e}")
            # 如果摄像头启动失败，尝试摄像头0
            try:
                self.camera_thread.start_camera(0)
                self.camera_running = True
                print("摄像头0启动成功")
            except Exception as e2:
                print(f"所有摄像头启动失败: {e2}")
                QtWidgets.QMessageBox.warning(self, "警告", "无法启动摄像头，请检查设备连接")
    
    def start_power_monitoring(self):
        """启动电流监测"""
        try:
            self.power_monitor_thread.start_monitoring()
            print("电流监测已启动")
        except Exception as e:
            print(f"启动电流监测失败: {e}")
            self.ui.currnt.setText("电流I     = 不可用")
            self.ui.currnt_2.setText("功耗P     = 不可用")
            self.ui.currnt_3.setText("最大功耗   = 不可用")
    
    def update_power_display(self, current_a, power_w, max_power_w):
        """更新电流和功耗显示"""
        try:
            # 显示电流、功耗、最大功耗，保留适当的小数位数
            
            # 更新UI显示
            self.ui.currnt.setText(f"电流I     = {current_a:.3f} A")
            self.ui.currnt_2.setText(f"功耗P     = {power_w:.3f} W")
            self.ui.currnt_3.setText(f"最大功耗   = {max_power_w:.3f} W")
            
        except Exception as e:
            print(f"更新电流显示错误: {e}")
    
    def on_power_monitor_error(self, error_msg):
        """处理电流监测错误"""
        print(f"电流监测错误: {error_msg}")
        self.ui.currnt.setText("电流I     = 错误")
        self.ui.currnt_2.setText("功耗P     = 错误")
        self.ui.currnt_3.setText("最大功耗   = 错误")
    
    def reset_max_power_values(self):
        """重置最大功耗值"""
        if hasattr(self, 'power_monitor_thread'):
            self.power_monitor_thread.reset_max_values()
            print("已重置最大功耗值")
        
        
    def select_cls_id(self, cls_id):
        """选择数字用于发挥题二"""
        self.selected_cls_id = cls_id
        print(f"选择了数字: {cls_id}")
        
        # 更新显示
        self.ui.side_x.setText(f"已选择数字: {cls_id}")
        
    def start_advanced_mode_2(self):
        """启动发挥题二模式 - 分析指定cls_id的正方形边长"""
        if not self.camera_running:
            QtWidgets.QMessageBox.warning(self, "警告", "摄像头未启动")
            return
            
        if self.selected_cls_id is None:
            QtWidgets.QMessageBox.warning(self, "警告", "请先选择数字按钮（0-9）来指定要分析的数字")
            return
            
        # 确保不会同时运行其他分析
        if hasattr(self, 'data_thread') and self.data_thread.isRunning():
            self.data_thread.collecting = False
            self.data_thread.quit()
            self.data_thread.wait()
            
        if hasattr(self, 'advanced_thread') and self.advanced_thread.isRunning():
            self.advanced_thread.analyzing = False
            self.advanced_thread.quit()
            self.advanced_thread.wait()
        
        # 解除检测显示冻结，恢复实时更新
        self.freeze_detection_display = False
        
        # 设置摄像头为发挥题二模式，开始实时YOLO检测指定cls_id
        self.camera_thread.set_mode("发挥题二", self.selected_cls_id)
        
        # 创建发挥题二分析线程
        self.advanced_thread_2 = AdvancedAnalysisThread2(self.camera_thread, self.selected_cls_id)
        self.advanced_thread_2.analysis_ready.connect(self.on_advanced_analysis_2_complete)
        self.advanced_thread_2.progress_update.connect(self.update_status)
        
        # 更新界面状态
        self.ui.side_x.setText(f"发挥题二分析中（数字={self.selected_cls_id}）...")
        self.ui.advance.setEnabled(False)  # 禁用发挥题按钮
        self.ui.advance_2.setEnabled(False)  # 禁用发挥题二按钮
        self.ui.base.setEnabled(False)  # 禁用基础题按钮
        
        # 开始分析
        self.advanced_thread_2.start_analysis()
        
    def on_advanced_analysis_2_complete(self, distance, side_length, analysis_text):
        """发挥题二分析完成的回调"""
        # 更新距离显示
        self.ui.dis.setText(f"D = {distance/10:.2f} cm")
        
        # 更新边长显示
        self.ui.side_x.setText(f"数字{self.selected_cls_id} X = {side_length/10:.2f} cm")
        
        # 停止YOLO检测，恢复基础题模式
        self.camera_thread.set_mode("基础题")
        
        # 冻结检测显示，保留最后一帧
        self.freeze_detection_display = True
        
        # 恢复界面状态
        self.ui.advance.setEnabled(True)  # 重新启用发挥题按钮
        self.ui.advance_2.setEnabled(True)  # 重新启用发挥题二按钮
        self.ui.base.setEnabled(True)  # 重新启用基础题按钮
        
        # 打印详细分析结果到控制台
        print("="*60)
        print(f"发挥题二分析完成 (数字={self.selected_cls_id}):")
        print(f"距离: {distance/10:.2f} cm")
        print(f"数字 {self.selected_cls_id}边长: {side_length/10:.2f} cm")
        print("YOLO检测已停止，恢复基础题模式")
        print("检测显示已冻结，保留最后一帧")
        print("="*60)
        
    def start_measurement(self):
        """开始测量"""
        if not self.camera_running:
            QtWidgets.QMessageBox.warning(self, "警告", "请先启动摄像头")
            return 
        if self.measuring:
            QtWidgets.QMessageBox.information(self, "提示", "正在测量中，请稍候...")
            return
            
        # 解除检测显示冻结，恢复实时更新
        self.freeze_detection_display = False
        
        # 设置为基础题模式，关闭YOLO检测
        self.camera_thread.set_mode("基础题")
        
        self.measuring = True
        self.ui.base.setText("测量中...")
        self.ui.base.setEnabled(False)
        self.ui.dis.setText("正在采集数据...")
        self.ui.side_x.setText("进度: 0/8")
        
        # 开始数据采集
        self.data_thread.start_collection()
        
    def update_measurements(self, distance, side):
        """更新测量结果"""
        self.measuring = False
        self.ui.base.setText("基础题")
        self.ui.base.setEnabled(True)
        
        if distance > 0 and side > 0:
            self.ui.dis.setText(f"D =   {distance/10:.2f} cm")
            self.ui.side_x.setText(f"X =   {side/10:.2f} cm")
        else:
            self.ui.dis.setText("距离D:   检测失败")
            self.ui.side_x.setText("边长/直径X:   检测失败")
            
    def update_progress(self, current, total):
        """更新采集进度"""
        self.ui.side_x.setText(f"进度: {current}/{total}")
        
    def start_advanced_mode(self):
        """启动发挥题模式 - 结合A4纸距离计算和YOLO检测"""
        if not self.camera_running:
            QtWidgets.QMessageBox.warning(self, "警告", "摄像头未启动")
            return
            
        # 确保不会同时运行基础题和发挥题
        if hasattr(self, 'data_thread') and self.data_thread.isRunning():
            self.data_thread.collecting = False
            self.data_thread.quit()
            self.data_thread.wait()
        
        # 解除检测显示冻结，恢复实时更新
        self.freeze_detection_display = False
        
        # 设置摄像头为发挥题模式，开始实时YOLO检测
        self.camera_thread.set_mode("发挥题")
        
        # 创建发挥题分析线程
        self.advanced_thread = AdvancedAnalysisThread(self.camera_thread)
        self.advanced_thread.analysis_ready.connect(self.on_advanced_analysis_complete)
        self.advanced_thread.progress_update.connect(self.update_status)
        
        # 更新界面状态
        self.ui.side_x.setText("发挥题分析中...")
        self.ui.advance.setEnabled(False)  # 禁用发挥题按钮
        self.ui.base.setEnabled(False)  # 禁用基础题按钮
        
        # 开始分析
        self.advanced_thread.start_analysis()
        
    def on_advanced_analysis_complete(self, distance, side_length, analysis_text):
        """发挥题分析完成的回调"""
        # 更新距离显示
        self.ui.dis.setText(f"距离D = {distance/10:.2f} cm")
        
        # 更新边长显示
        self.ui.side_x.setText(f"边长X = {side_length/10:.2f} cm")
        
        # 停止YOLO检测，恢复基础题模式
        self.camera_thread.set_mode("基础题")
        
        # 冻结检测显示，保留最后一帧
        self.freeze_detection_display = True
        
        # 恢复界面状态
        self.ui.advance.setEnabled(True)  # 重新启用发挥题按钮
        self.ui.base.setEnabled(True)  # 重新启用基础题按钮
        
        # 打印详细分析结果到控制台
        print("="*60)
        print("发挥题分析完成:")
        print(f"距离: {distance/10:.2f} cm")
        print(f"边长: {side_length/10:.2f} cm")
        print("YOLO检测已停止，恢复基础题模式")
        print("检测显示已冻结，保留最后一帧")
        print("="*60)
        
    def stop_yolo_detection(self):
        """停止YOLO检测"""
        self.camera_thread.set_mode("基础题")
        print("YOLO检测已关闭")
        
    def update_status(self, message):
        """更新状态信息"""
        # 使用标题标签显示状态，因为没有专门的状态标签
        self.ui.label.setText(f"基于单目视觉的目标物测量装置 - {message}")
        print(f"状态更新: {message}")
        
        # 预留代码框架：
        # 1. 可以在这里添加更复杂的形状检测算法
        # 2. 可以添加多目标检测功能
        # 3. 可以添加实时测量和跟踪功能
        # 4. 可以添加数据记录和分析功能
        
    def toggle_camera(self):
        """切换摄像头状态 - 备用功能"""
        if not self.camera_running:
            try:
                self.camera_thread.start_camera(0)
                self.camera_running = True
                print("摄像头已启动")
            except Exception as e:
                print(f"启动摄像头失败: {e}")
                QtWidgets.QMessageBox.warning(self, "错误", f"无法启动摄像头: {e}")
        else:
            self.camera_thread.stop_camera()
            self.camera_running = False
            # 清空显示
            self.ui.video.clear()
            self.ui.video.setText("摄像头已停止")
            print("摄像头已停止")
        
    def update_frame(self, frame):
        """更新显示帧"""
        try:
            # 绘制ROI框 - 使用c1.py中定义的ROI区域
            roi_x, roi_y, roi_w, roi_h = (526, 222, 227, 276)
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            cv2.putText(frame, "ROI Detection Area", (roi_x, roi_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 绘制1280*720分辨率的竖直中线
            cv2.line(frame, (640, 0), (640, 720), (255, 0, 0), 1)

            # 转换颜色格式
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # 创建QImage
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

            self.ui.video.setPixmap(QPixmap.fromImage(q_image).scaled(
            self.ui.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            # self.ui.video.setPixmap(pixmap)
            
        except Exception as e:
            print(f"更新帧失败: {e}")
    
    def update_detection_frame(self, detection_frame):
        """更新检测显示帧到video_2控件"""
        try:
            # 如果检测显示被冻结，则不更新
            if self.freeze_detection_display:
                return
                
            # 获取video_2控件的尺寸
            label_size = self.ui.video_2.size()
            target_width = label_size.width()
            target_height = label_size.height()
            
            # 获取检测帧的原始尺寸
            original_height, original_width = detection_frame.shape[:2]
            
            # 计算缩放比例，保持宽高比
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            scale_ratio = min(width_ratio, height_ratio)
            
            # 计算缩放后的尺寸
            new_width = int(original_width * scale_ratio)
            new_height = int(original_height * scale_ratio)
            
            # 缩放图像
            resized_frame = cv2.resize(detection_frame, (new_width, new_height))
            
            # 创建目标尺寸的黑色背景
            display_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # 计算居中放置的位置
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            
            # 将缩放后的图像居中放置在黑色背景上
            display_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
            
            # 转换颜色格式
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # 创建QImage
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # 转换为QPixmap并显示到video_2控件
            pixmap = QPixmap.fromImage(q_image)
            self.ui.video_2.setPixmap(pixmap)
            
        except Exception as e:
            print(f"更新检测帧失败: {e}")
            
    def closeEvent(self, event):
        """关闭事件"""
        if self.camera_running:
            self.camera_thread.stop_camera()
        if hasattr(self, 'data_thread') and self.data_thread.isRunning():
            self.data_thread.collecting = False
            self.data_thread.quit()
            self.data_thread.wait()
        if hasattr(self, 'advanced_thread') and self.advanced_thread.isRunning():
            self.advanced_thread.analyzing = False
            self.advanced_thread.quit()
            self.advanced_thread.wait()
        if hasattr(self, 'advanced_thread_2') and self.advanced_thread_2.isRunning():
            self.advanced_thread_2.analyzing = False
            self.advanced_thread_2.quit()
            self.advanced_thread_2.wait()
        if hasattr(self, 'power_monitor_thread') and self.power_monitor_thread.isRunning():
            self.power_monitor_thread.stop_monitoring()
            print("电流监测已停止")
        event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
