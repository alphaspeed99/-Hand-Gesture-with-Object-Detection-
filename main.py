import cv2
import mediapipe as mp
import pyautogui
import os
import datetime
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from subprocess import call

# Configuration Constants
GESTURE_FRAME_COUNT_THRESHOLD = 20
DISTANCE_CLICK_THRESHOLD = 0.03
DISTANCE_SCROLL_THRESHOLD = 0.10

# Initialize MediaPipe Hand modules
mp_hands = mp.solutions.hands
hands_cursor = mp_hands.Hands(max_num_hands=2)
hands_gesture = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Set up screen size for cursor movement
screen_width, screen_height = pyautogui.size()

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize gesture recognition variables
gesture_frames_detected = 0

detector = HandDetector(detectionCon=0.8, maxHands=1)  # Limit to detecting one hand
# Initialize button coordinates and sizes
button_width = 70
button_height = 30
button_margin = 5
undo_button_coords = (button_margin, button_margin)
photo_button_coords = (button_margin  + button_width +10, button_margin)
quit_button_coords = (button_margin + button_width +100, button_margin)
od_button_coords = (button_margin + button_width + 450, button_margin)  # Adjust the X coordinate as needed
vd_button_coords = (button_margin + button_width + 300, button_margin)  # Adjust the X coordinate as needed


border_color = (0, 255, 0)  # Green color for the border
border_thickness = 2

undo_button_clicked = False  # Flag to track button click
photo_button_clicked = False
quit_button_clicked = False
# od_button_coords = True
vd_button_clicked = False
left_click_pressed = False

# Initialize the list to store actions
actions = []

# Create a directory for saving photos if it doesn't exist
PHOTO_DIR = 'captured_photos'
if not os.path.exists(PHOTO_DIR):
    os.makedirs(PHOTO_DIR)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands, _ = detector.findHands(frame)

    # Process the frame for cursor movement
    cursor_results = hands_cursor.process(frame_rgb)

    # Process the frame for gesture recognition
    gesture_results = hands_gesture.process(frame_rgb)
    
    # Reset gesture recognition status
    is_gesture = False

    # Reset cursor movement status
    move_cursor = False

    # Draw buttons on the frame
    frame = cv2.rectangle(frame, (0, 0),  (640, 40), (225, 225, 225), -1)
    # cv2.putText(frame, "FILE", (50, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(frame, undo_button_coords, (undo_button_coords[0] + button_width, undo_button_coords[1] + button_height), (255, 0, 0), -1)
    cv2.putText(frame, "Undo", (undo_button_coords[0] + 5, undo_button_coords[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2 )
    # Draw border around the rectangle
    cv2.rectangle(frame, undo_button_coords, (undo_button_coords[0] + button_width, undo_button_coords[1] + button_height), (0, 0, 0), 2)  # Black border


    cv2.rectangle(frame, photo_button_coords , (photo_button_coords[0] + button_width, photo_button_coords[1] + button_height), (0, 255, 0), -1)
    cv2.putText(frame, "Photo", (photo_button_coords[0] + 5, photo_button_coords[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.rectangle(frame, photo_button_coords , (photo_button_coords[0] + button_width, photo_button_coords[1] + button_height), (0, 0, 0), 2)  # Black border
    
    
    cv2.rectangle(frame, quit_button_coords, (quit_button_coords[0] + button_width, quit_button_coords[1] + button_height), (0, 0, 255), -1)
    cv2.rectangle(frame, quit_button_coords, (quit_button_coords[0] + button_width, quit_button_coords[1] + button_height), (0, 0, 0), 2)  # Black border
    cv2.putText(frame, "Quit", (quit_button_coords[0] + 10, quit_button_coords[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw the "VD" button
    cv2.rectangle(frame, vd_button_coords, (vd_button_coords[0] + button_width, vd_button_coords[1] + button_height), (0, 0, 0), 3)  # Black border
    cv2.rectangle(frame, vd_button_coords, (vd_button_coords[0] + button_width, vd_button_coords[1] + button_height), (0, 0, 255), -1)  # Filled color
    cv2.putText(frame, "VD", (vd_button_coords[0] + 18, vd_button_coords[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    # Draw the OJD button
    cv2.rectangle(frame, od_button_coords, (od_button_coords[0] + button_width, od_button_coords[1] + button_height), (0, 0, 0), 3)  # Black border
    cv2.rectangle(frame, od_button_coords, (od_button_coords[0] + button_width, od_button_coords[1] + button_height), (255, 0, 0), -1)  # Filled color
    cv2.putText(frame, "OJD", (od_button_coords[0] + 18, od_button_coords[1] + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Add real-time date and time
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, current_time, (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if cursor_results.multi_hand_landmarks:
        for hand_landmarks in cursor_results.multi_hand_landmarks:
            index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger_landmark.x * screen_width)
            y = int(index_finger_landmark.y * screen_height)
            pyautogui.moveTo(x, y)
            move_cursor = True
    
    if move_cursor:
        # Process the frame for gesture recognition
        gesture_results = hands_cursor.process(frame_rgb)
        for landmarks in gesture_results.multi_hand_landmarks:
            # Handle click gesture
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            index_coords = np.array([index_finger_tip.x, index_finger_tip.y])
            middle_coords = np.array([middle_finger_tip.x, middle_finger_tip.y])
            
            finger_x, finger_y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)
            # delta_x, delta_y = finger_x - mouse_x, finger_y - mouse_y

            distance = np.linalg.norm(index_coords - middle_coords)
            

            # Check for left-click gesture (thumb and index finger pinched)
            thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x, thumb_y = int(thumb_tip.x * screen_width), int(thumb_tip.y * screen_height)

            if abs(finger_x - thumb_x) < 20 and abs(finger_y - thumb_y) < 20:
                if not left_click_pressed:
                    pyautogui.mouseDown()
                    left_click_pressed = True
            else:
                if left_click_pressed:
                    pyautogui.mouseUp()
                    left_click_pressed = False


            thumb_landmark = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_landmark = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            if distance < DISTANCE_CLICK_THRESHOLD:
                pyautogui.click()

            thumb_x = thumb_landmark.x * screen_width
            index_finger_x = index_finger_landmark.x * screen_width
            
            if thumb_x < index_finger_x:
                pyautogui.scroll(6)  # Scroll down
                print("scroll down")

            elif thumb_x > index_finger_x:
                pyautogui.scroll(-6)  # Scroll up
                print("scroll up")
            

            # Draw cursor and hand landmarks on the frame
            cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 0), cv2.FILLED)
            detector.findHands(frame, draw=False)  # Draw hand landmarks

    
    if gesture_results.multi_hand_landmarks:
        for landmarks in gesture_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            thumb_coords = np.array([thumb_tip.x, thumb_tip.y])
            pinky_coords = np.array([pinky_tip.x, pinky_tip.y])
            distance = np.linalg.norm(thumb_coords - pinky_coords)

            if distance < 0.3:
                is_gesture = True

    
    # Handle gesture recognition
    if is_gesture:
        gesture_frames_detected += 1
        if gesture_frames_detected >= GESTURE_FRAME_COUNT_THRESHOLD:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            photo_name = os.path.join(PHOTO_DIR, f'photo_{timestamp}.jpg')
            cv2.imwrite(photo_name, frame)
            print(f"Photo captured (Gesture Detected): {photo_name}")
            gesture_frames_detected = 0
    else:
        gesture_frames_detected = 0

    # Check for button clicks
    if cursor_results.multi_hand_landmarks:
        for hand_landmarks in cursor_results.multi_hand_landmarks:
            index_finger_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger_landmark.x * screen_width)
            y = int(index_finger_landmark.y * screen_height)

            # Check if buttons are clicked
            if undo_button_coords[0] < x < undo_button_coords[0] + button_width and \
               undo_button_coords[1] < y < undo_button_coords[1] + button_height:
                undo_button_clicked = True
                actions.append(frame.copy())

            elif photo_button_coords[0] < x < photo_button_coords[0] + button_width and \
                 photo_button_coords[1] < y < photo_button_coords[1] + button_height:
                photo_button_clicked = True

            elif quit_button_coords[0] < x < quit_button_coords[0] + button_width and \
                 quit_button_coords[1] < y < quit_button_coords[1] + button_height:
                quit_button_clicked = True

            elif vd_button_coords[0] < x < vd_button_coords[0] + button_width and \
                vd_button_coords[1] < y < vd_button_coords[1] + button_height:
                # vd_button_clicked = True
                cv2.waitKey(2000)  # Show the "Welcome" message for 2 seconds
                cap.release()  # Release the camera resources
                cv2.destroyAllWindows()
                def open_py_file():
                    call(["python","Virtual_drawing.py"])

                open_py_file()

                # exit()

            elif od_button_coords[0] < x < od_button_coords[0] + button_width and \
                 od_button_coords[1] < y < od_button_coords[1] + button_height:
                cv2.waitKey(2000)  # Show the "Welcome" message for 2 seconds
                cap.release()  # Release the camera resources
                cv2.destroyAllWindows()
                def open_py_file():
                    call(["python","objectnms.py"])

                open_py_file()

                exit()

    # Handle button actions
    if undo_button_clicked:
        if actions:
            print("Undo button clicked - Performing Undo Action")
            # Get the previous frame from the actions list
            frame = actions.pop()
            undo_button_clicked = False    # Handle button actions
    # if undo_button_clicked:
    #     print("Undo button clicked - Perform Undo Action")
    #     undo_button_clicked = False

    if photo_button_clicked:
        # Capture and save a photo
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        photo_name = os.path.join(PHOTO_DIR, f'photo_{timestamp}.jpg')
        cv2.imwrite(photo_name, frame)
        print(f"Photo captured: {photo_name}")
        photo_button_clicked = False

    if quit_button_clicked:
        print("Quit button clicked - Exiting the application")
        break



    
    # Display the frame
    cv2.imshow('Gesture Recognition and Cursor Control', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
