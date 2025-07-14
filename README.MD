# 🖐️ Hand Gesture Mouse Controller (with Scroll & Click)

Control your mouse using your hand gestures in real time with a webcam!

Built using Python, OpenCV, MediaPipe, and PyAutoGUI.

![Demo](assets/demo.gif) <!-- Replace with actual demo GIF or image -->

---

## ✨ Features

✅ Move mouse with index + middle fingers  
✅ Left Click using thumb + index pinch  
✅ Right Click using thumb + middle pinch  
✅ Scroll Up using "Thumbs Up" (all other fingers closed)  
✅ Scroll Down using "Thumbs Down" (all other fingers closed)  
✅ Pause all gestures by making a Fist  
✅ Visual feedback: displays current action in top-left corner

---

## 📦 Requirements

Python 3.7 or later.

Install all required dependencies using:

```bash
pip install -r requirements.txt
````

requirements.txt:

```txt
opencv-python
mediapipe
pyautogui
```

---

## 🚀 How to Run

1. Clone this repository:

```bash
git clone https://github.com/venkatram-2005/Mouse-Control-With-Guesture.git
cd Mouse-Control-With-Guesture
```

2. Create virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python main.py
```

Make sure your webcam is connected and functional.

---

## 🖐️ Supported Gestures

| Gesture                 | Action         | How To Perform                       |
| ----------------------- | -------------- | ------------------------------------ |
| ✌️ Index + Middle Up    | Move cursor    | Keep index & middle fingers up       |
| 🤏 Thumb + Index pinch  | Left Click     | Bring thumb and index tips together  |
| 🤏 Thumb + Middle pinch | Right Click    | Bring thumb and middle tips together |
| 👍 Thumb Up             | Scroll Up      | Thumb up, all other fingers closed   |
| 👎 Thumb Down           | Scroll Down    | Thumb down, all other fingers closed |
| ✊ Fist                  | Pause gestures | Make a fist to pause input           |

---

## 🛠️ Project Structure

```
hand-gesture-mouse/
├── main.py       # Main application code
├── requirements.txt       # Python dependencies
├── venv/
└── README.md              # This file
```

---

## 📝 License

This project is licensed under the MIT License. See below:

```
MIT License

Copyright (c) 2025 Venkatram Valluri

 
```
