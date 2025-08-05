import cv2
import numpy as np
import math

def find_longest_perpendicular_segment(input_image, angle_threshold=10, show_result=False):
    """
    寻找图像中最长的垂直线段
    
    参数:
        input_image: 输入图像，可以是文件路径(str)或numpy数组
        angle_threshold: 垂直角度阈值，默认10度
        show_result: 是否显示结果图像，默认False
    
    返回:
        tuple: (最长线段长度, 所有垂直线段信息, 标记后的图像, 简化图像)
        - max_length: 最长线段长度(像素)
        - perpendicular_segments: 所有垂直线段的信息列表
        - marked_img: 标记了线段的原始图像
        - simplified_img: 简化后的图像
    """
    try:
        # 处理输入图像
        if isinstance(input_image, str):
            # 如果是文件路径，读取图像
            img = cv2.imread(input_image)
            if img is None:
                raise ValueError(f"无法读取图像文件: {input_image}")
        else:
            # 如果是numpy数组，直接使用
            img = input_image.copy()
        
        # 灰度化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # 查找轮廓
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0, [], img, np.zeros_like(img)

        # 获取最大轮廓
        main_contour = max(contours, key=cv2.contourArea)

        # Douglas-Peucker算法简化
        epsilon = 0.01 * cv2.arcLength(main_contour, True)  # 精度参数
        approx = cv2.approxPolyDP(main_contour, epsilon, True)

        # 创建简化后的图像
        simplified_img = np.zeros_like(img)
        cv2.drawContours(simplified_img, [approx], -1, (0, 255, 0), 2)

        # 计算所有线段及其向量
        segments = []

        # 遍历所有相邻点对
        for i in range(len(approx)):
            # 获取当前点和下一个点（循环处理）
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % len(approx)][0]

            # 计算线段长度
            length = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

            # 计算线段向量
            vector = np.array([pt2[0] - pt1[0], pt2[1] - pt1[1]])

            # 存储线段信息
            segments.append({
                'points': (pt1, pt2),
                'length': length,
                'vector': vector,
                'index': i
            })

        # 寻找与两条其他线段垂直的最长线段
        max_length = 0
        longest_segment = None
        perpendicular_segments = []  # 存储所有满足垂直条件的线段

        # 检查每条线段是否与相邻线段垂直
        for i, seg in enumerate(segments):
            # 获取前一条线段
            prev_seg = segments[(i - 1) % len(segments)]
            # 获取后一条线段
            next_seg = segments[(i + 1) % len(segments)]

            # 计算与前一条线段的夹角
            dot_product_prev = np.dot(seg['vector'], prev_seg['vector'])
            mag_seg = np.linalg.norm(seg['vector'])
            mag_prev = np.linalg.norm(prev_seg['vector'])
            if mag_seg > 0 and mag_prev > 0:
                cos_angle_prev = dot_product_prev / (mag_seg * mag_prev)
                angle_prev = np.degrees(np.arccos(np.clip(cos_angle_prev, -1, 1)))
            else:
                angle_prev = 0

            # 计算与后一条线段的夹角
            dot_product_next = np.dot(seg['vector'], next_seg['vector'])
            mag_next = np.linalg.norm(next_seg['vector'])
            if mag_seg > 0 and mag_next > 0:
                cos_angle_next = dot_product_next / (mag_seg * mag_next)
                angle_next = np.degrees(np.arccos(np.clip(cos_angle_next, -1, 1)))
            else:
                angle_next = 0

            # 检查是否垂直（夹角接近90度）
            is_perpendicular = (abs(angle_prev - 90) < angle_threshold and
                                abs(angle_next - 90) < angle_threshold)

            # 如果是垂直线段，添加到候选列表
            if is_perpendicular:
                perpendicular_segments.append(seg)

                # 更新最长线段
                if seg['length'] > max_length:
                    max_length = seg['length']
                    longest_segment = seg

        # 创建标记后的图像副本
        marked_img = img.copy()
        marked_simplified = simplified_img.copy()

        # 在图像上标记最长垂直直线段
        if longest_segment:
            pt1, pt2 = longest_segment['points']
            # 在原始图像上绘制红色线段
            cv2.line(marked_img, tuple(pt1), tuple(pt2), (0, 0, 255), 3)
            # 在简化图像上绘制红色线段
            cv2.line(marked_simplified, tuple(pt1), tuple(pt2), (0, 0, 255), 3)

            # 添加文本标注
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            cv2.putText(marked_img, f"Length: {max_length:.2f}", (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(marked_simplified, f"Length: {max_length:.2f}", (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 标记相邻的垂直线段（黄色）
            idx = longest_segment['index']
            prev_seg = segments[(idx - 1) % len(segments)]
            next_seg = segments[(idx + 1) % len(segments)]

            # 绘制前一条线段
            prev_pt1, prev_pt2 = prev_seg['points']
            cv2.line(marked_img, tuple(prev_pt1), tuple(prev_pt2), (0, 255, 255), 2)
            cv2.line(marked_simplified, tuple(prev_pt1), tuple(prev_pt2), (0, 255, 255), 2)

            # 绘制后一条线段
            next_pt1, next_pt2 = next_seg['points']
            cv2.line(marked_img, tuple(next_pt1), tuple(next_pt2), (0, 255, 255), 2)
            cv2.line(marked_simplified, tuple(next_pt1), tuple(next_pt2), (0, 255, 255), 2)

            # 在交点处绘制标记
            cv2.circle(marked_img, tuple(pt1), 5, (255, 0, 0), -1)
            cv2.circle(marked_img, tuple(pt2), 5, (255, 0, 0), -1)
            cv2.circle(marked_simplified, tuple(pt1), 5, (255, 0, 0), -1)
            cv2.circle(marked_simplified, tuple(pt2), 5, (255, 0, 0), -1)

            # 打印结果信息
            print(f"最长垂直直线段长度: {max_length:.2f} 像素")
            print(f"找到 {len(perpendicular_segments)} 条垂直线段:")
            for seg in perpendicular_segments:
                print(f"  线段长度: {seg['length']:.2f} 像素")
        return max_length, perpendicular_segments, marked_img, marked_simplified
        
    except Exception as e:
        print(f"处理图像时发生错误: {e}")
        return 0, [], input_image if isinstance(input_image, np.ndarray) else np.zeros((100, 100, 3), dtype=np.uint8), np.zeros((100, 100, 3), dtype=np.uint8)

# def main():
#     """主函数 - 演示如何使用封装的函数"""
#     # 使用默认参数分析图像
#     max_length, perpendicular_segments, marked_img, simplified_img = find_longest_perpendicular_segment("2.png", show_result=True)
    
#     if max_length > 0:
#         print(f"\n分析完成！")
#         print(f"最长垂直线段长度: {max_length:.2f} 像素")
#         print(f"共找到 {len(perpendicular_segments)} 条垂直线段")
#     else:
#         print("未找到有效的垂直线段")

# if __name__ == "__main__":
#     main()