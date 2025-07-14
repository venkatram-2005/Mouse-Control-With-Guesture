import cv2
import mediapipe as mp
import pyautogui
import math
import time

screen_width, screen_height = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
frame_reduction = 100

click_flag = False
right_click_flag = False

last_action = ""
last_action_time = 0

paused = False

def fingers_up(lm_list):
    """
    Returns list of booleans: thumb, index, middle, ring, pinky
    True = finger is up/extended
    """
    fingers = []
    # Thumb: tip(4) x > ip(3) x (right hand assumption)
    fingers.append(lm_list[4][0] > lm_list[3][0])
    # For other fingers: tip y < pip y means finger up
    tips = [8, 12, 16, 20]
    for tip in tips:
        fingers.append(lm_list[tip][1] < lm_list[tip - 2][1])
    return fingers

def is_fist(lm_list):
    """Returns True if fist detected (all fingertips near palm)"""
    palm_y = lm_list[0][1]
    count = 0
    for tip_id in [4,8,12,16,20]:
        if lm_list[tip_id][1] > palm_y + 30:  # finger tip below wrist by margin
            count += 1
    return count >= 5

def thumb_direction(lm_list):
    """
    Returns 'up' if thumb points up,
            'down' if thumb points down,
            None otherwise
    """
    thumb_tip_y = lm_list[4][1]
    thumb_ip_y = lm_list[3][1]

    if thumb_ip_y - thumb_tip_y > 20:
        return 'up'
    elif thumb_tip_y - thumb_ip_y > 20:
        return 'down'
    else:
        return None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            if lm_list:
                # Pause all if fist
                if is_fist(lm_list):
                    if not paused:
                        paused = True
                        last_action = "Paused (Fist)"
                        last_action_time = time.time()
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    continue
                else:
                    if paused:
                        paused = False
                        last_action = "Resumed"
                        last_action_time = time.time()

                fingers = fingers_up(lm_list)
                thumb = fingers[0]
                others_closed = not any(fingers[1:])  # All fingers except thumb closed

                # Scroll only if thumb up/down and others closed
                direction = thumb_direction(lm_list)
                if thumb and others_closed and direction is not None:
                    if direction == 'up':
                        pyautogui.scroll(50)
                        last_action = "Scroll Up (Thumb Up)"
                    else:
                        pyautogui.scroll(-50)
                        last_action = "Scroll Down (Thumb Down)"
                    last_action_time = time.time()
                else:
                    # Normal gestures if not paused and not scrolling
                    x_index, y_index = lm_list[8]
                    x_thumb, y_thumb = lm_list[4]
                    x_middle, y_middle = lm_list[12]

                    # Move cursor only if index and middle fingers up
                    if fingers[1] and fingers[2]:
                        screen_x = int((x_index - frame_reduction) * screen_width / (w - 2 * frame_reduction))
                        screen_y = int((y_index - frame_reduction) * screen_height / (h - 2 * frame_reduction))
                        pyautogui.moveTo(screen_x, screen_y)

                    dist_thumb_index = math.hypot(x_thumb - x_index, y_thumb - y_index)
                    if dist_thumb_index < 40:
                        if not click_flag:
                            pyautogui.click()
                            click_flag = True
                            last_action = "Left Click"
                            last_action_time = time.time()
                    else:
                        click_flag = False

                    dist_thumb_middle = math.hypot(x_thumb - x_middle, y_thumb - y_middle)
                    if dist_thumb_middle < 40:
                        if not right_click_flag:
                            pyautogui.rightClick()
                            right_click_flag = True
                            last_action = "Right Click"
                            last_action_time = time.time()
                    else:
                        right_click_flag = False

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show last action on screen for 1.5 seconds
    if time.time() - last_action_time < 1.5:
        cv2.putText(frame, f"Action: {last_action}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
