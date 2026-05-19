"""
越线计数模块
满足任务二要求：虚拟线计数功能
"""

import cv2
import numpy as np
from collections import deque
from typing import Tuple, Optional

class LineCounter:
    """
    越线计数器
    
    原理：记录每个目标的中心点轨迹，当连续两帧中心点位于计数线两侧时，
    判定为越线，并根据跨越方向进行计数。
    """
    
    def __init__(self, line_points: Tuple[Tuple[int, int], Tuple[int, int]]):
        """
        初始化计数器
        
        Args:
            line_points: 线段端点，格式 ((x1,y1), (x2,y2))
        """
        self.line_start = np.array(line_points[0], dtype=np.float32)
        self.line_end = np.array(line_points[1], dtype=np.float32)
        
        # 已计数目标集合（防止重复计数）
        self.crossed_ids = set()
        
        # 计数统计
        self.count_up = 0      # 正向计数
        self.count_down = 0    # 反向计数
        
        # 目标历史轨迹
        self.trajectories = {}
        
        # 计数线向量和法向量
        self.line_vec = self.line_end - self.line_start
        self.line_normal = np.array([-self.line_vec[1], self.line_vec[0]])
        norm = np.linalg.norm(self.line_normal)
        if norm > 0:
            self.line_normal = self.line_normal / norm
    
    def get_side(self, point: np.ndarray) -> float:
        """
        判断点在线段的哪一侧
        
        使用叉积计算：
        - 正值：一侧
        - 负值：另一侧
        - 0：在线段上
        """
        vec = point - self.line_start
        return np.cross(self.line_vec, vec)
    
    def check_crossing(self, track_id: int, 
                       prev_center: np.ndarray, 
                       curr_center: np.ndarray) -> Optional[str]:
        """
        检查目标是否越线
        
        Args:
            track_id: 跟踪ID
            prev_center: 上一帧中心点
            curr_center: 当前帧中心点
            
        Returns:
            'up': 正向跨越 | 'down': 反向跨越 | None: 未跨越或已计数
        """
        if track_id in self.crossed_ids:
            return None
        
        prev_side = self.get_side(prev_center)
        curr_side = self.get_side(curr_center)
        
        # 判断是否跨越（两侧异号）
        if prev_side * curr_side < 0:
            self.crossed_ids.add(track_id)
            
            # 根据跨越方向计数
            movement = curr_center - prev_center
            dot = np.dot(movement, self.line_normal)
            
            if dot > 0:
                self.count_up += 1
                return 'up'
            else:
                self.count_down += 1
                return 'down'
        
        return None
    
    def update_trajectory(self, track_id: int, center: np.ndarray, max_len: int = 30):
        """更新目标轨迹历史"""
        if track_id not in self.trajectories:
            self.trajectories[track_id] = deque(maxlen=max_len)
        self.trajectories[track_id].append(center.copy())
    
    def get_trajectory(self, track_id: int) -> list:
        """获取目标轨迹"""
        if track_id in self.trajectories:
            return list(self.trajectories[track_id])
        return []
    
    def reset_count(self):
        """重置计数器"""
        self.crossed_ids.clear()
        self.count_up = 0
        self.count_down = 0
    
    def draw_line(self, frame: np.ndarray) -> np.ndarray:
        """
        在图像上绘制计数线和统计信息
        """
        # 绘制计数线
        pt1 = tuple(self.line_start.astype(int))
        pt2 = tuple(self.line_end.astype(int))
        cv2.line(frame, pt1, pt2, (0, 0, 255), 3)
        
        # 绘制方向指示箭头
        mid_point = (self.line_start + self.line_end) / 2
        arrow_end = mid_point + self.line_normal * 30
        cv2.arrowedLine(frame, 
                       tuple(mid_point.astype(int)), 
                       tuple(arrow_end.astype(int)), 
                       (0, 255, 0), 2, tipLength=0.3)
        
        # 显示统计信息
        h, w = frame.shape[:2]
        
        # 背景框
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # 文字
        cv2.putText(frame, f"Up: {self.count_up}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Down: {self.count_down}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Total: {self.count_up + self.count_down}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def draw_trajectory(self, frame: np.ndarray, track_id: int, 
                       color: Tuple[int, int, int]) -> np.ndarray:
        """绘制单个目标的运动轨迹"""
        trajectory = self.get_trajectory(track_id)
        if len(trajectory) < 2:
            return frame
        
        for i in range(1, len(trajectory)):
            pt1 = tuple(trajectory[i-1].astype(int))
            pt2 = tuple(trajectory[i].astype(int))
            cv2.line(frame, pt1, pt2, color, 2)
        
        return frame