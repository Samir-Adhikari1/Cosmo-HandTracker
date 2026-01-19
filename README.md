# 🤖 Cosmo-HandTracker 🖐️
Real-time hand tracking system using Python, AI, and computer vision. Includes hardware integration and software implementation for gesture recognition and servo control.

[![Platform](https://img.shields.io/badge/Platform-Arduino-blue)](https://www.arduino.cc/)
[![Language](https://img.shields.io/badge/Language-Python-orange)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-PCA9685-green)](https://learn.adafruit.com/16-channel-pwm-servo-driver)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](https://opensource.org/licenses/MIT)

**Author:** Cosmo  
**System Stack:** Python (Computer Vision) · Arduino · PCA9685 PWM Driver  
**Domain:** Robotics · Human–Machine Interaction · Embedded Systems  

---

## 🔍 Overview

**Cosmo-HandTracker** is a real-time **hand tracking–based servo control system** that translates natural human hand motion into precise mechanical actuation. Using a standard webcam, the system extracts **high-resolution hand kinematics** by computing **14 biomechanical joint angles** and transmits them to an Arduino via serial communication.

A **PCA9685 PWM driver** controls **five servo motors**, each mapped to a finger.  
The system is designed to be **modular, scalable, and hardware-agnostic**, making it suitable for robotics, prosthetics research, and teleoperation systems.

---

## ⭐ Key Contributions

- End-to-end pipeline from **vision-based perception to embedded actuation**
- **14-joint kinematic hand model** derived from anatomical structure
- Continuous, low-latency serial streaming for smooth motion
- Hardware abstraction via PCA9685 (supports up to 16 actuators)
- Design suitable for **research, academic evaluation, and real-world prototyping**

---

## 🧠 System Architecture

### 🔁 High-Level Data Flow

1. Webcam captures live video frames  
2. Python detects 21 hand landmarks (MediaPipe)  
3. Joint angles are computed using geometric relationships  
4. Angle data is serialized as CSV  
5. Data is transmitted over USB serial  
6. Arduino parses and validates the stream  
7. PCA9685 generates precise PWM signals  
8. Servo motors replicate human finger motion  

This architecture enables **continuous, proportional, and intuitive control**.

---

## ✋ Hand Kinematic Model

The human hand is modeled using joint-angle representations derived from landmark triplets:

- **Thumb:** 2 joints  
- **Index Finger:** 3 joints  
- **Middle Finger:** 3 joints  
- **Ring Finger:** 3 joints  
- **Pinky Finger:** 3 joints  

**Total:** 14 joint angles

### 🎯 Servo Mapping Strategy

- One representative joint angle per finger is mapped to a servo motor  
- Remaining angles are preserved for future multi-joint actuation  
- This balances **biomechanical realism** with **hardware simplicity**

---

## 🔧 Hardware Requirements

- Arduino UNO (or compatible microcontroller)
- PCA9685 16-Channel PWM Servo Driver (I²C)
- 5 × Servo Motors (SG90 / MG90 or equivalent)
- External 5–6 V power supply (mandatory for servos)
- Webcam
- USB cable and jumper wires

> ⚠️ **Safety Note:** Servo motors must be powered externally. Drawing servo current from the Arduino 5 V rail may permanently damage the board.

---

## 🔌 Hardware Connections

### PCA9685 ↔ Arduino (I²C)

| PCA9685 Pin | Arduino UNO |
|------------|-------------|
| VCC        | 5V          |
| GND        | GND         |
| SDA        | A4          |
| SCL        | A5          |

### Servo Channel Assignment

| Finger  | PCA9685 Channel |
|--------|------------------|
| Thumb  | Channel 0 |
| Index  | Channel 1 |
| Middle | Channel 2 |
| Ring   | Channel 3 |
| Pinky | Channel 4 |

---

## 💻 Software Requirements

### Python (PC Side)

- Python 3.9+
- OpenCV
- MediaPipe
- NumPy
- PySerial

Installation:
```bash
pip install opencv-python mediapipe numpy pyserial
