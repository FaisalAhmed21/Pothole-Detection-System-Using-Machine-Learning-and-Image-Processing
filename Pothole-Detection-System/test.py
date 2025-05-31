from ultralytics import YOLO
import cv2
import numpy as np
import time
import math
import os
import datetime
from collections import deque, defaultdict

model = YOLO("best_02.pt")
class_names = model.names
cap = cv2.VideoCapture('p.mp4')
count = 0

small_threshold = 5000
large_threshold = 15000

small_color = (0, 255, 0)
medium_color = (0, 165, 255)
large_color = (0, 0, 255)

max_disappeared = 30
next_pothole_id = 0
pothole_trackers = {}

pothole_count = {"Small": 0, "Medium": 0, "Large": 0}
final_pothole_count = {"Small": 0, "Medium": 0, "Large": 0}
detected_areas = []
risk_levels = {"Low": 0, "Medium": 0, "High": 0}

all_detected_potholes = []
pothole_locations = []
frame_count = 0
video_duration = 0

obstruction_map = np.zeros((500, 800), dtype=np.uint8)

fps = 0
processing_times = deque(maxlen=30)

def calculate_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    else:
        x, y, w, h = cv2.boundingRect(contour)
        return (x + w // 2, y + h // 2)

def calculate_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def get_risk_level(size_category, position_y, road_height):
    road_center = road_height / 2
    center_factor = 1 - (abs(position_y - road_center) / road_center)
    
    if size_category == "Large":
        risk_base = 2
    elif size_category == "Medium":
        risk_base = 1
    else:
        risk_base = 0
    
    risk_adjusted = risk_base + center_factor
    
    if risk_adjusted >= 2:
        return "High", (0, 0, 255)
    elif risk_adjusted >= 1:
        return "Medium", (0, 165, 255)
    else:
        return "Low", (0, 255, 0)

def find_safe_path(obstruction_map, w, h):
    safe_cols = []
    lower_region = obstruction_map[h//2:, :]
    slice_width = w // 5
    for i in range(5):
        start_col = i * slice_width
        end_col = (i + 1) * slice_width
        slice_sum = np.sum(lower_region[:, start_col:end_col])
        safe_cols.append((slice_sum, (start_col + end_col) // 2))
    safe_cols.sort(key=lambda x: x[0])
    return [col[1] for col in safe_cols[:3]]

def update_trackers(new_potholes):
    global next_pothole_id, pothole_trackers, all_detected_potholes, final_pothole_count
    
    for pothole_id in list(pothole_trackers.keys()):
        pothole_trackers[pothole_id]["disappeared"] += 1
        if pothole_trackers[pothole_id]["disappeared"] > max_disappeared:
            if pothole_id not in [p["id"] for p in all_detected_potholes]:
                all_detected_potholes.append({
                    "id": pothole_id,
                    "size": pothole_trackers[pothole_id]["final_size"],
                    "area": pothole_trackers[pothole_id]["max_area"],
                    "risk": pothole_trackers[pothole_id]["final_risk"],
                    "frames_visible": max_disappeared - pothole_trackers[pothole_id]["disappeared"],
                    "position": pothole_trackers[pothole_id]["centroid"]
                })
                final_pothole_count[pothole_trackers[pothole_id]["final_size"]] += 1
            del pothole_trackers[pothole_id]
    
    if len(new_potholes) == 0:
        return
    
    if len(pothole_trackers) == 0:
        for i in range(len(new_potholes)):
            pothole_trackers[next_pothole_id] = {
                "centroid": new_potholes[i]["centroid"],
                "disappeared": 0,
                "size": new_potholes[i]["size"],
                "contour": new_potholes[i]["contour"],
                "area": new_potholes[i]["area"],
                "risk": new_potholes[i]["risk"],
                "max_area": new_potholes[i]["area"],
                "final_size": new_potholes[i]["size"],
                "final_risk": new_potholes[i]["risk"]
            }
            next_pothole_id += 1
    else:
        tracker_ids = list(pothole_trackers.keys())
        tracker_centroids = [pothole_trackers[tid]["centroid"] for tid in tracker_ids]
        distance_matrix = []
        for new_pothole in new_potholes:
            distances = []
            for tracker_centroid in tracker_centroids:
                d = calculate_distance(new_pothole["centroid"], tracker_centroid)
                distances.append(d)
            distance_matrix.append(distances)
        
        used_trackers = set()
        used_potholes = set()
        
        while True:
            min_distance = float("inf")
            min_pothole_idx = -1
            min_tracker_idx = -1
            
            for i in range(len(new_potholes)):
                if i in used_potholes:
                    continue
                for j in range(len(tracker_ids)):
                    if j in used_trackers:
                        continue
                    if distance_matrix[i][j] < min_distance:
                        min_distance = distance_matrix[i][j]
                        min_pothole_idx = i
                        min_tracker_idx = j
            
            if min_pothole_idx == -1 or min_distance > 50:
                break
                
            tracker_id = tracker_ids[min_tracker_idx]
            pothole_trackers[tracker_id]["centroid"] = new_potholes[min_pothole_idx]["centroid"]
            pothole_trackers[tracker_id]["disappeared"] = 0
            pothole_trackers[tracker_id]["size"] = new_potholes[min_pothole_idx]["size"]
            pothole_trackers[tracker_id]["contour"] = new_potholes[min_pothole_idx]["contour"]
            pothole_trackers[tracker_id]["area"] = new_potholes[min_pothole_idx]["area"]
            pothole_trackers[tracker_id]["risk"] = new_potholes[min_pothole_idx]["risk"]
            
            if new_potholes[min_pothole_idx]["area"] > pothole_trackers[tracker_id]["max_area"]:
                pothole_trackers[tracker_id]["max_area"] = new_potholes[min_pothole_idx]["area"]
                pothole_trackers[tracker_id]["final_size"] = new_potholes[min_pothole_idx]["size"]
                pothole_trackers[tracker_id]["final_risk"] = new_potholes[min_pothole_idx]["risk"]
            
            used_trackers.add(min_tracker_idx)
            used_potholes.add(min_pothole_idx)
        
        for i in range(len(new_potholes)):
            if i not in used_potholes:
                pothole_trackers[next_pothole_id] = {
                    "centroid": new_potholes[i]["centroid"],
                    "disappeared": 0,
                    "size": new_potholes[i]["size"],
                    "contour": new_potholes[i]["contour"],
                    "area": new_potholes[i]["area"],
                    "risk": new_potholes[i]["risk"],
                    "max_area": new_potholes[i]["area"],
                    "final_size": new_potholes[i]["size"],
                    "final_risk": new_potholes[i]["risk"]
                }
                next_pothole_id += 1

fps_video = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = total_frames / fps_video if fps_video > 0 else 0

layout_width = 1200
layout_height = 600
    
while True:
    start_time = time.time()
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame_count += 1
    img = cv2.resize(img, (800, 500))
    h, w, _ = img.shape
    obstruction_map = np.zeros((h, w), dtype=np.uint8)
    pothole_count = {"Small": 0, "Medium": 0, "Large": 0}
    risk_levels = {"Low": 0, "Medium": 0, "High": 0}
    results = model.predict(img)
    overlay = img.copy()
    new_potholes = []

    for r in results:
        boxes = r.boxes
        masks = r.masks
        
    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                centroid = calculate_centroid(contour)
                
                if area < small_threshold:
                    size_category = "Small"
                elif area > large_threshold:
                    size_category = "Large"
                else:
                    size_category = "Medium"
                
                risk_level, risk_color = get_risk_level(size_category, centroid[1], h)
                pothole_count[size_category] += 1
                detected_areas.append(area)
                risk_levels[risk_level] += 1
                cv2.drawContours(obstruction_map, [contour], -1, 255, -1)
                new_potholes.append({
                    "centroid": centroid,
                    "size": size_category,
                    "contour": contour,
                    "area": area,
                    "risk": risk_level
                })
                pothole_locations.append(centroid)
                d = int(box.cls)
                c = class_names[d]
    
    update_trackers(new_potholes)
    
    for pothole_id, data in pothole_trackers.items():
        if data["disappeared"] == 0:
            contour = data["contour"]
            final_size = data["final_size"]
            centroid = data["centroid"]
            final_risk = data["final_risk"]
            
            # Only mark potholes in bottom 25% of screen
            if centroid[1] >= h * 0.75:
                if final_risk == "High":
                    color = (0, 0, 255)
                elif final_risk == "Medium":
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 0)
                
                cv2.drawContours(overlay, [contour], -1, color, -1)
                cv2.polylines(img, [contour], True, color=color, thickness=3)
            
            x, y, width, height = cv2.boundingRect(contour)
            label = f"ID:{pothole_id} {final_size} (Risk: {final_risk})"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(img, centroid, 4, (255, 0, 255), -1)
    
    safe_cols = find_safe_path(obstruction_map, w, h)
    for col in safe_cols:
        start_point = (col, h)
        end_point = (col, h - 100)
        cv2.line(img, start_point, end_point, (255, 255, 0), 2)
        cv2.circle(img, (col, h - 110), 5, (255, 255, 0), -1)

    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    end_time = time.time()
    processing_time = end_time - start_time
    processing_times.append(processing_time)
    
    if len(processing_times) > 0:
        avg_time = sum(processing_times) / len(processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

    # Create layout with video covering the entire screen
    layout = np.zeros((layout_height, layout_width, 3), dtype=np.uint8)
    
    # Place video on the entire layout (full screen)
    layout[0:500, 0:1200] = cv2.resize(img, (1200, 500))
    
    # Draw statistics in left corner (top-left)
    cv2.rectangle(layout, (10, 10), (300, 160), (30, 30, 30), -1)
    cv2.rectangle(layout, (10, 10), (300, 160), (100, 100, 100), 2)
    
    # Add dashboard title
    cv2.putText(layout, "Pothole Statistics", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add pothole counts
    y_pos = 70
    cv2.putText(layout, "Current Frame:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(layout, "Unique Potholes:", (160, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    y_pos += 25
    cv2.putText(layout, f"Small: {pothole_count['Small']}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, small_color, 2)
    cv2.putText(layout, f"Small: {final_pothole_count['Small']}", (160, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, small_color, 2)
    
    y_pos += 25
    cv2.putText(layout, f"Medium: {pothole_count['Medium']}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, medium_color, 2)
    cv2.putText(layout, f"Medium: {final_pothole_count['Medium']}", (160, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, medium_color, 2)
    
    y_pos += 25
    cv2.putText(layout, f"Large: {pothole_count['Large']}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, large_color, 2)
    cv2.putText(layout, f"Large: {final_pothole_count['Large']}", (160, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, large_color, 2)
    
    # Display FPS in top-right corner
    cv2.putText(layout, f"FPS: {fps:.1f}", (layout_width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display safe path indicator at bottom
    cv2.putText(layout, "Safe Driving Paths", (500, layout_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Add instruction at the bottom
    cv2.putText(layout, "Press 'q' to exit and generate report", (layout_width - 350, layout_height - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('Pothole Detection System', layout)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for pothole_id, data in pothole_trackers.items():
    if pothole_id not in [p["id"] for p in all_detected_potholes]:
        all_detected_potholes.append({
            "id": pothole_id,
            "size": data["final_size"],
            "area": data["max_area"],
            "risk": data["final_risk"],
            "frames_visible": max_disappeared - data["disappeared"],
            "position": data["centroid"]
        })
        final_pothole_count[data["final_size"]] += 1

hotspots = []
if len(pothole_locations) > 0:
    for i, loc in enumerate(pothole_locations):
        nearby_count = 0
        for other_loc in pothole_locations:
            if calculate_distance(loc, other_loc) < 50:
                nearby_count += 1
        if nearby_count >= 3 and loc not in [h["center"] for h in hotspots]:
            hotspots.append({
                "center": loc,
                "count": nearby_count
            })
hotspots.sort(key=lambda x: x["count"], reverse=True)
avg_area = sum(detected_areas) / len(detected_areas) if len(detected_areas) > 0 else 0
high_risk_count = sum(1 for p in all_detected_potholes if p["risk"] == "High")
medium_risk_count = sum(1 for p in all_detected_potholes if p["risk"] == "Medium")
severity_rating = min(10, (medium_risk_count * 0.5 + high_risk_count * 1.0) / max(1, len(all_detected_potholes)) * 10)
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
report_filename = f"pothole_analysis_report_{current_time}.txt"

with open(report_filename, "w") as report_file:
    report_file.write("="*80 + "\n")
    report_file.write(f"POTHOLE DETECTION ANALYSIS REPORT\n")
    report_file.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_file.write("="*80 + "\n\n")
    report_file.write("VIDEO INFORMATION\n")
    report_file.write("-"*80 + "\n")
    report_file.write(f"Filename: {os.path.basename('p.mp4')}\n")
    report_file.write(f"Duration: {video_duration:.2f} seconds\n")
    report_file.write(f"Frames analyzed: {frame_count}\n")
    report_file.write(f"Model used: YOLOv8-seg (best_02.pt)\n\n")
    report_file.write("SUMMARY STATISTICS\n")
    report_file.write("-"*80 + "\n")
    report_file.write(f"Total unique potholes detected: {sum(final_pothole_count.values())}\n")
    report_file.write(f"Pothole size distribution:\n")
    report_file.write(f"  - Small: {final_pothole_count['Small']}\n")
    report_file.write(f"  - Medium: {final_pothole_count['Medium']}\n")
    report_file.write(f"  - Large: {final_pothole_count['Large']}\n\n")
    final_risk_levels = {"Low": 0, "Medium": 0, "High": 0}
    for pothole in all_detected_potholes:
        final_risk_levels[pothole["risk"]] += 1
    report_file.write(f"Risk level distribution:\n")
    report_file.write(f"  - Low risk: {final_risk_levels['Low']}\n")
    report_file.write(f"  - Medium risk: {final_risk_levels['Medium']}\n")
    report_file.write(f"  - High risk: {final_risk_levels['High']}\n\n")
    if len(all_detected_potholes) > 0:
        final_avg_area = sum(p["area"] for p in all_detected_potholes) / len(all_detected_potholes)
        report_file.write(f"Average pothole area: {final_avg_area:.2f} square pixels\n")
    else:
        report_file.write(f"Average pothole area: 0 square pixels\n")
    report_file.write(f"Overall road condition severity rating (0-10): {severity_rating:.1f}\n\n")
    report_file.write("HOTSPOT ANALYSIS\n")
    report_file.write("-"*80 + "\n")
    if len(hotspots) > 0:
        report_file.write(f"Identified {len(hotspots)} hotspot areas:\n")
        for i, hotspot in enumerate(hotspots[:5]):
            report_file.write(f"  {i+1}. Location: x={hotspot['center'][0]}, y={hotspot['center'][1]} - {hotspot['count']} potholes\n")
    else:
        report_file.write("No significant hotspots identified.\n")
    report_file.write("\n")
    report_file.write("DETAILED POTHOLE INFORMATION\n")
    report_file.write("-"*80 + "\n")
    sorted_potholes = sorted(all_detected_potholes, key=lambda x: {"High": 2, "Medium": 1, "Low": 0}[x["risk"]], reverse=True)
    for i, pothole in enumerate(sorted_potholes):
        report_file.write(f"Pothole #{i+1} (ID: {pothole['id']}):\n")
        report_file.write(f"  - Size: {pothole['size']}\n")
        report_file.write(f"  - Area: {pothole['area']:.2f} pixels\n")
        report_file.write(f"  - Risk: {pothole['risk']}\n")
        report_file.write(f"  - Position: x={pothole['position'][0]}, y={pothole['position'][1]}\n")
        report_file.write(f"  - Frames visible: {pothole['frames_visible']}\n")
        report_file.write("\n")
    report_file.write("RECOMMENDATIONS\n")
    report_file.write("-"*80 + "\n")
    if severity_rating >= 7:
        report_file.write("URGENT ATTENTION REQUIRED: Significant pothole damage detected\n")
    elif severity_rating >= 4:
        report_file.write("MODERATE ATTENTION NEEDED: Moderate pothole damage detected\n")
    else:
        report_file.write("MINOR ATTENTION SUGGESTED: Minimal pothole damage\n")

print(f"Analysis complete. Report saved to {report_filename}")

cap.release()
cv2.destroyAllWindows()