import cv2
import mediapipe as mp
import math
import mido
from mido import Message

# Open a virtual MIDI output port
outport = mido.open_output('Gesture MIDI Output', virtual=True)


# --- Helpers -
def distance(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def finger_up(hand_landmarks, finger_tip, finger_dip):
    """Return True if finger is extended (tip higher than dip)"""
    return hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_dip].y

def send_midi(gesture):
    gesture_map = {
        "🤏 Pinch": 60,     # C4
        "☝️ Pointing": 62,  # D4
        "✌️ Peace": 64,     # E4
        "🖐 Open Hand": 65, # F4
        "✊ Fist": 67,       # G4
        "👍 Thumbs Up": 69, # A4
        "🤘 Hell yeah!!": 71 # B4
    }

    if gesture in gesture_map:
        note = gesture_map[gesture]
        msg = Message('note_on', note=note, velocity=100)
        outport.send(msg)
        print(f"Sent MIDI Note ON: {note}")

        import time
        time.sleep(0.2)
        outport.send(Message('note_off', note=note, velocity=100))


# --- Mediapipe setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

print("Gesture Detector running... Press ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Key landmarks
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            # Finger up detection (compare tip to DIP joint)
            index_up = finger_up(hand_landmarks, 8, 6)
            middle_up = finger_up(hand_landmarks, 12, 10)
            ring_up = finger_up(hand_landmarks, 16, 14)
            pinky_up = finger_up(hand_landmarks, 20, 18)

            # Distances
            pinch_dist = distance(thumb_tip, index_tip)

            # --- Gesture rules ---
            if pinch_dist < 0.05:
                gesture_text = "🤏 Pinch"
            elif index_up and not (middle_up or ring_up or pinky_up):
                gesture_text = "☝️ Pointing"
            elif index_up and middle_up and not (ring_up or pinky_up):
                gesture_text = "✌️ Peace"
            elif all([index_up, middle_up, ring_up, pinky_up]):
                gesture_text = "🖐 Open Hand"
            elif not any([index_up, middle_up, ring_up, pinky_up]):
                gesture_text = "✊ Fist"
            elif pinky_up and not (index_up or middle_up or ring_up):
                gesture_text = "🤟 Pinky"
            elif thumb_tip.x < hand_landmarks.landmark[3].x and all([index_up, middle_up, ring_up, pinky_up]):
                gesture_text = "👍 Thumbs Up"
            elif pinky_up and index_up and not (middle_up or ring_up):
                gesture_text = "🤘 Hell yeah!!"
            else:
                gesture_text = "🤚 Unknown"
                
            if gesture_text != "🤚 Unknown" and gesture_text != "":
                send_midi(gesture_text)

            # Show gesture text
            #cv2.putText(frame, gesture_text, (50, 50),
                        #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
