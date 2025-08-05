import cv2
import numpy as np

# 实际A4尺寸
A4_WIDTH_MM = 170
A4_HEIGHT_MM = 257


# 假设焦距值
FOCAL_HEIGHT_PX = 747.73109
FOCAL_WIDTH_PX = 741.18396
#cap = cv2.VideoCapture("/dev/video0")

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
def order_points(pts):
    """对A4角点排序为：左上，右上，右下，左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]     # 左上
    rect[2] = pts[np.argmax(s)]     # 右下
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def find_a4_contour(image):
    """寻找矩形框，总计两个矩形框"""
    roi_x, roi_y, roi_w, roi_h = (526, 222, 227, 276)
    roi_image = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

    cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged=cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow("Edges", edged)  # 注释掉显示窗口
    
    # 先筛选出四边形轮廓
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:  # 只保留四边形
            valid_contours.append(cnt)
            #print(f"Contour Area: {area}, Approx Length: {len(approx_1)}")
    
    # 对筛选后的四边形轮廓进行排序
    if len(valid_contours) < 2:
        return None
    
    sorted_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    target = sorted_contours[1]
    # cv2.drawContours(roi_image, [target], -1, (0, 255, 0), 2)  # 绘制目标轮廓
    # cv2.drawContours(roi_image, [sorted_contours[0]], -1, (255, 255, 0), 2)  # 绘制最大轮廓
    
    # cv2.imshow("Target Contour", roi_image)  # 显示目标轮廓
    #对target轮廓进行多边形逼近
    epsilon = 0.02 * cv2.arcLength(target, True)
    approx = cv2.approxPolyDP(target, epsilon, True)
    # cv2.imshow("Approx Contour", roi_image)  # 显示逼近后的轮廓       
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.35 < aspect_ratio < 0.75:
            # 对target进行get_topdown_view
            warped = get_topdown_view(roi_image, order_points(approx.reshape(4, 2)))
            #显示变换后的宽高
            cv2.putText(image, f"W: {warped.shape[1]}, H: {warped.shape[0]}", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.imshow("Topdown View", warped)  # 注释掉显示窗口
            cv2.drawContours(roi_image, [approx], -1, (0, 255, 0), 2)
            return warped
    return None

def get_topdown_view(image, pts):
    """透视变换,将A4纸变为标准正视图，保持原始像素尺寸"""
    # 计算A4纸在图像中的实际像素尺寸
    width_top = np.linalg.norm(pts[1] - pts[0])      # 上边长度
    width_bottom = np.linalg.norm(pts[2] - pts[3])   # 下边长度
    height_left = np.linalg.norm(pts[3] - pts[0])    # 左边长度
    height_right = np.linalg.norm(pts[2] - pts[1])   # 右边长度
    
    #对高取平均值，宽通过A4纸比例计算
    max_height = int((height_left + height_right) / 2)
    max_width = int(max_height * (A4_WIDTH_MM / A4_HEIGHT_MM))

    
    # 目标点坐标，保持检测到的实际像素尺寸
    dst = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

def estimate_distance(focal_length_px, real_height_mm, image_height_px):
    """计算摄像头距离"""
    distance = (real_height_mm * focal_length_px) / image_height_px
    return distance

def find_shape(warped, distance_x, distance_y, fx, fy):
    """在变换后的图像中寻找形状，并计算实际边长"""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    ret,bin =cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # edged = cv2.Canny(blurred, 50, 150)
    # cv2.imshow("Edges", bin)  # 注释掉显示窗口

    contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(warped, contours, -1, (0, 255, 0), 2)  # 绘制所有轮廓
    # cv2.imshow("Contours", warped)  # 显示轮廓
    
    # 初始化返回值
    avg_side = 0
    detected_shapes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        print(area)
        if area >500:  # 忽略小轮廓
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            #框出轮廓
            cv2.drawContours(warped, [cnt], -1, (0, 255, 0), 2)
            print(f"Contour Area: {area}, Approx Length: {len(approx)}")
            if len(approx) == 3:
                # 计算三角形每条边的实际长度
                sides_mm = []
                for i in range(3):
                    p1 = approx[i][0]
                    p2 = approx[(i+1)%3][0]
                    dx = (p2[0] - p1[0]) * distance_x / fx
                    dy = (p2[1] - p1[1]) * distance_y / fy
                    side_length_mm = np.sqrt(dx**2 + dy**2)
                    sides_mm.append(side_length_mm)
                
                # 计算平均边长（像素和毫米）
                side1 = np.linalg.norm(approx[0] - approx[1])
                side2 = np.linalg.norm(approx[1] - approx[2])
                side3 = np.linalg.norm(approx[2] - approx[0])
                current_avg_side = (side1 + side2 + side3) / 3
                side_mm = np.mean(sides_mm) + 3.7
                detected_shapes.append({"type": "Triangle", "size": current_avg_side, "area": area, "side_mm": side_mm})
                
            elif len(approx) == 4:
                # 计算矩形每条边的实际长度
                sides_mm = []
                for i in range(4):
                    p1 = approx[i][0]
                    p2 = approx[(i+1)%4][0]
                    dx = (p2[0] - p1[0]) * distance_x / fx
                    dy = (p2[1] - p1[1]) * distance_y / fy
                    side_length_mm = np.sqrt(dx**2 + dy**2)
                    sides_mm.append(side_length_mm)
                
                # 计算平均边长（像素和毫米）
                side1 = np.linalg.norm(approx[0] - approx[1])
                side2 = np.linalg.norm(approx[1] - approx[2])
                side3 = np.linalg.norm(approx[2] - approx[3])
                side4 = np.linalg.norm(approx[3] - approx[0])
                current_avg_side = (side1 + side2 + side3 + side4) / 4
                side_mm = np.mean(sides_mm) + 4
                detected_shapes.append({"type": "Rectangle", "size": current_avg_side, "area": area, "side_mm": side_mm})
                
            elif len(approx) > 4:
                # 对于圆形，计算直径的实际长度
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                # 计算水平和垂直方向的直径
                dx_mm = (2 * radius) * distance_x / fx
                dy_mm = (2 * radius) * distance_y / fy
                # 使用平均值作为直径
                diameter_mm = (dx_mm + dy_mm) / 2
                
                current_avg_side = 2 * radius
                side_mm = diameter_mm
                detected_shapes.append({"type": "Circle", "size": current_avg_side, "area": area, "side_mm": side_mm})
        
    # 如果检测到形状，返回最小的一个
    if detected_shapes:
        smallest_shape = min(detected_shapes, key=lambda x: x["area"])
        avg_side = smallest_shape["size"]
        side_mm = smallest_shape["side_mm"]
        return avg_side, side_mm
    
    # cv2.imshow("Detected Shapes", warped)  # 注释掉显示窗口
    return 0, 0

'''
if __name__ == "__main__":
    # 移除主循环，因为现在由PyQt5控制
    while True:
        ret, frame = cap.read()
        if not ret:
            break  

        warped = find_a4_contour(frame)
        # cv2.imshow("A4 Contour", warped)  # 显示A4轮廓
        if warped is not None:
            distance_1 = estimate_distance(FOCAL_HEIGHT_PX, A4_HEIGHT_MM, warped.shape[0])
            distance_2 = estimate_distance(FOCAL_WIDTH_PX, A4_WIDTH_MM, warped.shape[1])
            avg_side_px, side_mm = find_shape(warped, distance_1, distance_2, FOCAL_WIDTH_PX, FOCAL_HEIGHT_PX)
            
            if avg_side_px > 0:  # 确保检测到了有效的形状
                cv2.putText(frame, f"Side: {avg_side_px:.2f} px ({side_mm:.2f} mm)", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Distance: {distance_1/10:.2f} cm", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Width Distance: {distance_2/10:.2f} cm", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "No shape detected", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(20)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 
 '''
