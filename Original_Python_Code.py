import cv2
import mediapipe as mp
import math
import serial
import time
import threading
import numpy as np
import serial.tools.list_ports
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    def __init__(self):
        self.SERIAL_PORT = 'COM7'
        self.BAUDRATE = 9600
        self.MAX_HANDS = 1
        self.SEND_INTERVAL = 5
        self.SMOOTHING_FACTOR = 0.8
        self.GESTURE_CONFIDENCE_THRESHOLD = 0.7
        self.ENABLE_OFFENSIVE_GESTURES = False  
        self.FRAME_SKIP_THRESHOLD = 20 
        self.FRAME_SKIP_FACTOR = 2 
        self.HAND_LOST_TIMEOUT = 2.0  

config = Config()

GESTURES_FILE = 'gestures.json'
if os.path.exists(GESTURES_FILE):
    with open(GESTURES_FILE, 'r') as f:
        gestures_data = json.load(f)
    GESTURES = gestures_data['gestures']
    THRESHOLDS = gestures_data['thresholds']
else:
    logging.warning(f"Gestures file '{GESTURES_FILE}' not found. Creating default gestures file.")
    GESTURES = {
        "Fist": {"condition": "extended_count == 0 and not thumb_ext", "confidence": 0.9, "offensive": False},
        "Open Hand": {"condition": "extended_count == 4 and thumb_ext", "confidence": 0.9, "offensive": False},
        "Point": {"condition": "extended_count == 1 and index_ext and not thumb_ext", "confidence": 0.8, "offensive": False},
        "Peace": {"condition": "extended_count == 2 and index_ext and middle_ext and not thumb_ext", "confidence": 0.8, "offensive": False},
        "Thumbs Up": {"condition": "thumb_ext and extended_count == 0", "confidence": 0.8, "offensive": False},
        "OK": {"condition": "thumb_index_dist < 0.05 and extended_count == 0", "confidence": 0.7, "offensive": False},
        "Middle Finger": {"condition": "extended_count == 1 and middle_ext and not thumb_ext and not index_ext and not ring_ext and not pinky_ext", "confidence": 0.8, "offensive": True}
    }
    THRESHOLDS = {
        "angle_validation_min": 0,
        "angle_validation_max": 180,
        "thumb_extension_threshold": 0.02
    }
    default_data = {"gestures": GESTURES, "thresholds": THRESHOLDS}
    with open(GESTURES_FILE, 'w') as f:
        json.dump(default_data, f, indent=4)
    logging.info(f"Default gestures file created at '{GESTURES_FILE}'.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=config.MAX_HANDS,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

LANDMARKS = {
    'wrist': 0, 'thumb_cmc': 1, 'thumb_mcp': 2, 'thumb_ip': 3, 'thumb_tip': 4,
    'index_mcp': 5, 'index_pip': 6, 'index_dip': 7, 'index_tip': 8,
    'middle_mcp': 9, 'middle_pip': 10, 'middle_dip': 11, 'middle_tip': 12,
    'ring_mcp': 13, 'ring_pip': 14, 'ring_dip': 15, 'ring_tip': 16,
    'pinky_mcp': 17, 'pinky_pip': 18, 'pinky_dip': 19, 'pinky_tip': 20
}

joint_definitions = [
    (LANDMARKS['thumb_cmc'], LANDMARKS['thumb_mcp'], LANDMARKS['thumb_ip']),
    (LANDMARKS['thumb_mcp'], LANDMARKS['thumb_ip'], LANDMARKS['thumb_tip']),
    (LANDMARKS['wrist'], LANDMARKS['index_mcp'], LANDMARKS['index_pip']),
    (LANDMARKS['index_mcp'], LANDMARKS['index_pip'], LANDMARKS['index_dip']),
    (LANDMARKS['index_pip'], LANDMARKS['index_dip'], LANDMARKS['index_tip']),
    (LANDMARKS['wrist'], LANDMARKS['middle_mcp'], LANDMARKS['middle_pip']),
    (LANDMARKS['middle_mcp'], LANDMARKS['middle_pip'], LANDMARKS['middle_dip']),
    (LANDMARKS['middle_pip'], LANDMARKS['middle_dip'], LANDMARKS['middle_tip']),
    (LANDMARKS['wrist'], LANDMARKS['ring_mcp'], LANDMARKS['ring_pip']),
    (LANDMARKS['ring_mcp'], LANDMARKS['ring_pip'], LANDMARKS['ring_dip']),
    (LANDMARKS['ring_pip'], LANDMARKS['ring_dip'], LANDMARKS['ring_tip']),
    (LANDMARKS['wrist'], LANDMARKS['pinky_mcp'], LANDMARKS['pinky_pip']),
    (LANDMARKS['pinky_mcp'], LANDMARKS['pinky_pip'], LANDMARKS['pinky_dip']),
    (LANDMARKS['pinky_pip'], LANDMARKS['pinky_dip'], LANDMARKS['pinky_tip'])
]

def calculate_angle(a, b, c):
    """Calculate angle at point b between points a, b, c."""
    ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]]
    dot = sum(x * y for x, y in zip(ba, bc))
    mag_ba = math.sqrt(sum(x**2 for x in ba))
    mag_bc = math.sqrt(sum(x**2 for x in bc))
    if mag_ba == 0 or mag_bc == 0:
        return 0.0
    angle_rad = math.acos(max(-1, min(1, dot / (mag_ba * mag_bc))))
    return math.degrees(angle_rad)

def validate_angles(angles):
    """Validate angles for anomalies."""
    for i, a in enumerate(angles):
        if not (THRESHOLDS['angle_validation_min'] <= a <= THRESHOLDS['angle_validation_max']):
            logging.warning(f"Invalid angle at index {i}: {a:.2f}")
            return False
    return True

class GestureRecognizer:
    def __init__(self):
        pass

    def is_finger_extended(self, landmarks, finger_tip, finger_pip, finger_mcp, angles, is_right_hand=True):
        """Check if a finger is extended, using position and angle for thumb."""
        if finger_tip == LANDMARKS['thumb_tip']:
            tip = landmarks[finger_tip]
            mcp = landmarks[finger_mcp]
            direction = 1 if is_right_hand else -1
            position_check = (tip.x - mcp.x) * direction > THRESHOLDS['thumb_extension_threshold']
            angle_check = angles[0] > 30  
            return position_check and angle_check
        else:
            return landmarks[finger_tip].y < landmarks[finger_pip].y

    def recognize(self, landmarks, angles, is_right_hand=True):
        """Recognize gesture with dynamic loading and confidence."""
        thumb_ext = self.is_finger_extended(landmarks, LANDMARKS['thumb_tip'], LANDMARKS['thumb_ip'], LANDMARKS['thumb_mcp'], angles, is_right_hand)
        index_ext = self.is_finger_extended(landmarks, LANDMARKS['index_tip'], LANDMARKS['index_pip'], LANDMARKS['index_mcp'], angles, is_right_hand)
        middle_ext = self.is_finger_extended(landmarks, LANDMARKS['middle_tip'], LANDMARKS['middle_pip'], LANDMARKS['middle_mcp'], angles, is_right_hand)
        ring_ext = self.is_finger_extended(landmarks, LANDMARKS['ring_tip'], LANDMARKS['ring_pip'], LANDMARKS['ring_mcp'], angles, is_right_hand)
        pinky_ext = self.is_finger_extended(landmarks, LANDMARKS['pinky_tip'], LANDMARKS['pinky_pip'], LANDMARKS['pinky_mcp'], angles, is_right_hand)

        extended_fingers = [index_ext, middle_ext, ring_ext, pinky_ext]
        extended_count = sum(extended_fingers)
        thumb_index_dist = math.sqrt((landmarks[LANDMARKS['thumb_tip']].x - landmarks[LANDMARKS['index_tip']].x)**2 +
                                     (landmarks[LANDMARKS['thumb_tip']].y - landmarks[LANDMARKS['index_tip']].y)**2)

        for name, data in GESTURES.items():
            if data.get('offensive', False) and not config.ENABLE_OFFENSIVE_GESTURES:
                continue
            condition = data['condition']
            try:
                if eval(condition, {"__builtins__": {}}, {
                    'extended_count': extended_count, 'thumb_ext': thumb_ext, 'index_ext': index_ext,
                    'middle_ext': middle_ext, 'ring_ext': ring_ext, 'pinky_ext': pinky_ext,
                    'thumb_index_dist': thumb_index_dist
                }):
                    conf = data['confidence']
                    if conf >= config.GESTURE_CONFIDENCE_THRESHOLD:
                        return name, conf
            except Exception as e:
                logging.error(f"Error evaluating gesture '{name}': {e}")
        return "Unknown", 0.0

def init_serial(port=None, baudrate=config.BAUDRATE):
    """Initialize serial with auto-detection."""
    if not port:
        ports = serial.tools.list_ports.comports()
        arduino_ports = [p.device for p in ports if 'Arduino' in p.description or 'USB' in p.description]
        port = arduino_ports[0] if arduino_ports else config.SERIAL_PORT
    try:
        arduino = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        logging.info(f"Arduino connected on {port}.")
        return arduino
    except serial.SerialException as e:
        logging.error(f"Could not connect to Arduino on {port}: {e}")
        return None

arduino = init_serial()
serial_queue = []
gesture_recognizer = GestureRecognizer()

def serial_worker():
    while True:
        if serial_queue and arduino:
            try:
                data = serial_queue.pop(0)
                arduino.write(data.encode())
                logging.debug(f"Sent to Arduino: {data.strip()}")
            except serial.SerialException as e:
                logging.error("Failed to send data to Arduino.")
        time.sleep(0.01)

threading.Thread(target=serial_worker, daemon=True).start()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logging.error("Could not open camera.")
    exit()

logging.info("Camera opened. Press 'q' to quit.")
frame_count = 0
prev_angles = None
prev_time = time.time()
skip_frame = False
hand_last_seen = time.time()  

while True:
    if skip_frame:
        cap.read()  
        skip_frame = False
        continue

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    if fps < config.FRAME_SKIP_THRESHOLD:
        skip_frame = True 

    if results.multi_hand_landmarks and results.multi_handedness:
        hand_last_seen = current_time  
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            is_right_hand = handedness.classification[0].label == 'Right'
            landmarks = hand_landmarks.landmark

            angles = []
            for p1, p2, p3 in joint_definitions:
                angle = calculate_angle([landmarks[p1].x, landmarks[p1].y, landmarks[p1].z],
                                        [landmarks[p2].x, landmarks[p2].y, landmarks[p2].z],
                                        [landmarks[p3].x, landmarks[p3].y, landmarks[p3].z])
                angles.append(angle)

            angles = [max(0, min(180, a)) for a in angles]
            if not validate_angles(angles):
                continue  

            if prev_angles:
                angles = [config.SMOOTHING_FACTOR * p + (1 - config.SMOOTHING_FACTOR) * c for p, c in zip(prev_angles, angles)]
            prev_angles = angles.copy()

            gesture, confidence = gesture_recognizer.recognize(landmarks, angles, is_right_hand)

            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in landmarks]
            y_coords = [int(lm.y * h) for lm in landmarks]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            cv2.rectangle(frame, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {gesture} ({'Right' if is_right_hand else 'Left'}) Conf: {confidence:.2f}",
                        (x_min - 20, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            angle_text = ", ".join(f"{a:.1f}" for a in angles[:5])
            cv2.putText(frame, f"Angles: {angle_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            frame_count += 1
            if frame_count % config.SEND_INTERVAL == 0:
                data = ",".join(f"{a:.2f}" for a in angles) + "\n"
                serial_queue.append(data)
    else:
        if current_time - hand_last_seen > config.HAND_LOST_TIMEOUT:
            logging.info("Hand lost for too long, resetting tracking.")
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=config.MAX_HANDS,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)  
            prev_angles = None  
            hand_last_seen = current_time
        cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if arduino:
    arduino.close()
cap.release()

cv2.destroyAllWindows()
