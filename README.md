# 🤖 Hand-Gesture-Controlled-Robotic-Hand 🖐️

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

**Cosmo-HandTracker** is a real-time **vision-driven hand tracking and servo actuation system** that converts natural human hand motion into precise mechanical movement. Using a standard webcam, the system extracts **high-resolution hand kinematic data** by computing **14 biomechanically inspired joint angles**, which are transmitted to an Arduino microcontroller via serial communication.

A **PCA9685 16-channel PWM driver** controls **five servo motors**, each mapped to an individual finger. The system architecture is **modular, scalable, and hardware-agnostic**, making it suitable for robotics research, prosthetics development, and human–machine interaction (HMI) experimentation.

---

## ⭐ Key Contributions

* Complete pipeline from **computer vision–based perception to embedded actuation**
* **14-joint kinematic hand model** derived from anatomical finger structure
* Continuous, low-latency serial data streaming for smooth and proportional motion
* Hardware abstraction using PCA9685 (supports up to 16 actuators)
* Design optimized for **academic evaluation, research prototyping, and real-world deployment**

---

## 🧠 System Architecture

### 🔁 High-Level Data Flow

1. Webcam captures live video frames
2. Python processes frames and detects **21 hand landmarks** using MediaPipe
3. Joint angles are computed using geometric relationships between landmarks
4. Angle data is serialized in CSV format
5. Data is transmitted over USB serial communication
6. Arduino parses and validates incoming data
7. PCA9685 generates high-precision PWM signals
8. Servo motors replicate human finger motion in real time

This pipeline enables **continuous, intuitive, and proportional control** with minimal latency.

---

## ✋ Hand Kinematic Model

The human hand is represented using a joint-angle–based kinematic model derived from landmark triplets:

* **Thumb:** 2 joint angles
* **Index Finger:** 3 joint angles
* **Middle Finger:** 3 joint angles
* **Ring Finger:** 3 joint angles
* **Pinky Finger:** 3 joint angles

**Total:** 14 joint angles

### 🎯 Servo Mapping Strategy

* One representative joint angle per finger is mapped to a corresponding servo motor
* Remaining joint angles are preserved for future multi-joint or tendon-driven actuation
* This strategy balances **biomechanical fidelity** with **hardware simplicity**

---

## 📦 System Requirements

### 🧩 Hardware Components

* **Arduino UNO** (or compatible microcontroller)
* **PCA9685 16-Channel PWM Servo Driver** (I²C interface)
* **5 × Micro Servo Motors (MG90S or equivalent metal-gear servos)**
* **External 5–6 V DC Power Adapter** (dedicated servo power supply)
* **Jumper Wires** (male–female / male–male as required)
* **3D-Printed Robotic Hand Structure**
* **Servo Mounting Accessories** (horns, screws, linkages)
* **Screwdriver Set**
* **Power Socket / Distribution Module**

### 💻 Computing Requirements (Host Machine)

* **Laptop or PC with AMD Ryzen 4000 Series processor or higher**
* **Efficient motherboard and thermal design** (required for sustained real-time computer vision processing)
* **Minimum 8 GB RAM** (16 GB recommended)
* **USB 2.0 / 3.0 port** for Arduino communication
* **Webcam** (720p minimum, 1080p recommended)

> ⚠️ **Performance Note:** Real-time hand tracking using MediaPipe and OpenCV is computationally intensive. Systems with weak cooling or inefficient motherboard power delivery may experience excessive thermal throttling and fan noise ("jet-engine" behavior), resulting in dropped frames and unstable servo response.**

> ⚠️ **Safety Notice:** Servo motors must be powered using an external supply. Supplying servo current directly from the Arduino 5 V rail may cause voltage drops or permanent damage to the microcontroller.

---

## 🔌 Hardware Connections

### 🔋 Power Distribution Strategy (Recommended)

During robotic hand assembly, **PCA9685 should be used as the primary power distribution unit for servo motors**, not the Arduino.

* **Servo power (V+) must be supplied directly to the PCA9685** using an external 5–6 V adapter
* **Arduino is used only for logic-level control and data communication**, not for powering servos
* This separation prevents voltage drops, electrical noise, and microcontroller resets

> ⚠️ **Critical Warning:** Do **NOT** power servo motors from Arduino GPIO pins or the Arduino 5 V rail. This can cause overheating, unstable motion, or permanent hardware damage.

### PCA9685 ↔ Arduino (I²C + Logic Power)

| PCA9685 Pin | Arduino UNO |
| ----------- | ----------- |
| VCC         | 5V (Logic)  |
| GND         | GND         |
| SDA         | A4          |
| SCL         | A5          |

### PCA9685 Servo Power Input

| PCA9685 Pin | Connection             |
| ----------- | ---------------------- |
| V+          | External 5–6 V Adapter |
| GND         | Common Ground          |

### Servo Channel Assignment

| Finger | PCA9685 Channel |
| ------ | --------------- |
| Thumb  | Channel 0       |
| Index  | Channel 1       |
| Middle | Channel 2       |
| Ring   | Channel 3       |
| Pinky  | Channel 4       |

---

## 💻 Software Requirements

### 🖥️ PC (Python Environment)

* Python 3.9 or later
* OpenCV
* MediaPipe
* NumPy
* PySerial

Installation:

```bash
pip install opencv-python mediapipe numpy pyserial
```

---

## 🧪 Application Domains

* Robotic hand and gripper control
* Prosthetics and rehabilitation research
* Teleoperation systems
* Human–robot interaction (HRI) studies
* Educational robotics and embedded systems learning

---

## 📄 License

This project is released under the **MIT License**, allowing free use, modification, and distribution for academic and commercial purposes.
