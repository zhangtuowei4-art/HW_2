"""
视频多目标跟踪主程序
满足任务二要求：
- 视频流检测与多目标跟踪
- 输出 BoundingBox 与 Tracking ID
- 越线计数
- 遮挡分析
"""

import cv2
import torch
import argparse
import numpy as np
from ultralytics import YOLO
from line_counter import LineCounter

class VideoTracker:
    def __init__(self, 
                 model_path: str = 'runs/train/vehicle_detection_v2/weights/best.pt',
                 tracker_config: str = 'config/bytetrack.yaml',
                 conf_threshold: float = 0.3,
                 iou_threshold: float = 0.5,
                 line_points=None):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model on {self.device}...")
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.line_counter = LineCounter(line_points) if line_points else None
        self.show_trajectory = True
        
        # 用于遮挡分析：记录每一帧的跟踪结果
        self.frame_results = []
    
    def _get_color(self, track_id: int):
        """根据 track_id 生成稳定的颜色"""
        np.random.seed(track_id * 31)
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    def process_frame(self, frame, frame_idx: int = 0):
        """
        处理单帧图像
    
        Args:
            frame: 输入图像
            frame_idx: 帧索引（用于遮挡分析）
            """
        if frame_idx == 1 and not hasattr(self, '_config_printed'):
            self._config_printed = True
            print("\n" + "="*70)
            print("🔧 ByteTrack 实际运行配置")
            print("="*70)
        
            # 等待 tracker 初始化完成
            import time
            for _ in range(10):
                if hasattr(self.model, 'tracker') and self.model.tracker is not None:
                    break
                time.sleep(0.01)
            
            # 方法1：通过 tracker.args 获取
            if hasattr(self.model, 'tracker') and self.model.tracker:
                if hasattr(self.model.tracker, 'args'):
                    args = self.model.tracker.args
                    print("\n📊 Tracker 参数:")
                    print(f"  tracker_type: {getattr(args, 'tracker_type', 'N/A')}")
                    print(f"  track_high_thresh: {getattr(args, 'track_high_thresh', 'N/A')}")
                    print(f"  track_low_thresh: {getattr(args, 'track_low_thresh', 'N/A')}")
                    print(f"  new_track_thresh: {getattr(args, 'new_track_thresh', 'N/A')}")
                    print(f"  track_buffer: {getattr(args, 'track_buffer', 'N/A')}")
                    print(f"  match_thresh: {getattr(args, 'match_thresh', 'N/A')}")
                    print(f"  fuse_score: {getattr(args, 'fuse_score', 'N/A')}")
                
                # 方法2：打印 tracker 的所有相关属性
                print("\n📋 所有 tracker 属性:")
                for attr in dir(self.model.tracker):
                    if not attr.startswith('_') and not callable(getattr(self.model.tracker, attr)):
                        val = getattr(self.model.tracker, attr)
                        if isinstance(val, (int, float, bool, str)):
                            print(f"  {attr}: {val}")
            
            # 打印配置文件路径
            import os
            print(f"\n📁 配置文件设置:")
            print(f"  tracker_config: {self.tracker_config}")
            
            # 检查配置文件是否存在
            if os.path.exists(self.tracker_config):
                print(f"  ✅ 配置文件存在: {os.path.abspath(self.tracker_config)}")
                print(f"\n📄 配置文件内容:")
                with open(self.tracker_config, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            print(f"    {line}")
            else:
                print(f"  ❌ 配置文件不存在: {os.path.abspath(self.tracker_config)}")
                print(f"  将使用 Ultralytics 默认配置")
                
                # 查找默认配置
                import ultralytics
                base = os.path.dirname(ultralytics.__file__)
                default_path = os.path.join(base, 'cfg', 'trackers', 'bytetrack.yaml')
                if os.path.exists(default_path):
                    print(f"\n📄 Ultralytics 默认配置 ({default_path}):")
                    with open(default_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                print(f"    {line}")
            
            print("="*70 + "\n")
    
        # ========== 原有代码开始 ==========
        results = self.model.track(frame, persist=True, tracker=self.tracker_config,
                                   conf=self.conf_threshold, iou=self.iou_threshold,
                                   device=self.device, verbose=False)
        
        detection_results = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confs):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                center = np.array([(x1 + x2)/2, (y1 + y2)/2])
            
                detection_results.append({
                    'frame': frame_idx,
                    'track_id': track_id,
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'confidence': float(conf)
                })
                
                if self.line_counter:
                    self.line_counter.update_trajectory(track_id, center)
                    traj = self.line_counter.get_trajectory(track_id)
                    if len(traj) >= 2:
                        crossing = self.line_counter.check_crossing(track_id, traj[-2], center)
                        if crossing:
                            print(f"Frame {frame_idx}: Vehicle {track_id} crossed {crossing}")
                
                # 可视化
                color = self._get_color(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID:{track_id} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, tuple(center.astype(int)), 3, (0, 0, 255), -1)
                
                if self.line_counter and self.show_trajectory:
                    frame = self.line_counter.draw_trajectory(frame, track_id, color)
        
        if self.line_counter:
            frame = self.line_counter.draw_line(frame)
    
        # 记录结果用于遮挡分析
        self.frame_results.append({
            'frame': frame_idx,
            'objects': detection_results
        })
        
        return frame
    
    def process_video(self, video_path, output_path=None):
        """
        处理视频文件
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # 初始化视频写入器
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        fps_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 处理帧
            start_time = cv2.getTickCount()
            processed_frame = self.process_frame(frame, frame_count)
            end_time = cv2.getTickCount()
            
            # 计算 FPS
            fps_current = cv2.getTickFrequency() / (end_time - start_time)
            fps_list.append(fps_current)
            
            # 显示信息
            cv2.putText(processed_frame, f"FPS: {fps_current:.1f}", 
                       (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, (0, 255, 0), 2)
            cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}", 
                       (width - 200, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)
            
            # 显示
            cv2.imshow('Vehicle Tracking', processed_frame)
            
            # 保存
            if out:
                out.write(processed_frame)
            
            # 退出条件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 进度显示
            if frame_count % 100 == 0:
                avg_fps = sum(fps_list[-100:]) / len(fps_list[-100:]) if fps_list else 0
                print(f"Processed: {frame_count}/{total_frames} frames, FPS: {avg_fps:.1f}")
        
        # 清理资源
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # 打印统计信息
        print(f"\n{'='*50}")
        print(f"处理统计")
        print(f"{'='*50}")
        print(f"处理帧数: {frame_count}")
        if fps_list:
            print(f"平均 FPS: {sum(fps_list) / len(fps_list):.2f}")
        
        if self.line_counter:
            print(f"越线计数总计: {self.line_counter.count_up + self.line_counter.count_down}")
            print(f"  向上 (Up): {self.line_counter.count_up}")
            print(f"  向下 (Down): {self.line_counter.count_down}")
        else:
            print("未配置计数线（添加 --line 参数启用）")
        print(f"{'='*50}")
    
    def analyze_occlusion(self, start_frame, end_frame):
        """
        遮挡分析：提取指定帧范围的结果
        """
        frames_data = [r for r in self.frame_results if start_frame <= r['frame'] <= end_frame]
        
        print(f"\n{'='*50}")
        print(f"遮挡分析 (帧 {start_frame} - {end_frame})")
        print(f"{'='*50}")
        
        for data in frames_data:
            track_ids = [obj['track_id'] for obj in data['objects']]
            print(f"Frame {data['frame']}: 检测到 {len(data['objects'])} 个目标, IDs: {track_ids}")
        
        return frames_data

    def process_camera(self, camera_id: int = 0):
        """处理摄像头实时流"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera: {camera_id}")
        
        print("Camera started. Press 'q' to quit, 'r' to reset counter, 't' to toggle trajectory")
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            processed_frame = self.process_frame(frame, frame_idx)
            
            h, w = processed_frame.shape[:2]
            cv2.putText(processed_frame, "q: quit | r: reset | t: trajectory", 
                       (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)
            
            cv2.imshow('Vehicle Tracking (Camera)', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and self.line_counter:
                self.line_counter.reset_count()
                print("Counter reset")
            elif key == ord('t'):
                self.show_trajectory = not self.show_trajectory
                print(f"Trajectory display: {self.show_trajectory}")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Vehicle Tracking with YOLOv8 + ByteTrack')
    parser.add_argument('--source', type=str, required=True,
                       help='视频文件路径或摄像头ID (如 0)')
    parser.add_argument('--model', type=str, default='runs/train/vehicle_detection_v2/weights/best.pt',
                       help='模型权重路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出视频路径')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IOU阈值')
    parser.add_argument('--line', type=str, default=None,
                       help='计数线坐标，格式: "x1,y1,x2,y2"')
    
    args = parser.parse_args()
    
    # 解析计数线坐标
    line_points = None
    if args.line:
        coords = list(map(int, args.line.split(',')))
        if len(coords) == 4:
            line_points = ((coords[0], coords[1]), (coords[2], coords[3]))
            print(f"Line set: {line_points}")
    
    # 创建跟踪器
    tracker = VideoTracker(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        line_points=line_points
    )
    
    # 判断输入类型
    if args.source.isdigit():
        tracker.process_camera(int(args.source))
    else:
        tracker.process_video(args.source, args.output)
        
        # 遮挡分析示例（可根据实际帧号修改）
        # frames = tracker.analyze_occlusion(230, 270)
        # for f in frames:
        #     print(f"Frame {f['frame']}: {f['objects']}")

if __name__ == '__main__':
    main()