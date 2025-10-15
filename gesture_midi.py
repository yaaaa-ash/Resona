import cv2
import mediapipe as mp
import mido
import math

# ---- MIDI setup ----
MIDI_PORT_NAME = 'GestureSound 1'
outport = mido.open_output(MIDI_PORT_NAME)

def send_note_on(note, velocity=127, channel=0):
    outport.send(mido.Message('note_on', note=int(note), velocity=int(velocity), channel=channel))

def send_note_off(note, velocity=0, channel=0):
    outport.send(mido.Message('note_off', note=int(note), velocity=int(velocity), channel=channel))

# ---- MediaPipe setup ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---- Right-hand note mapping ----
finger_notes = {8: 60, 12: 64, 16: 67, 20: 69}  # Index, Middle, Ring, Pinky = C E G A
finger_states = {tip_id: False for tip_id in finger_notes.keys()}

# ---- Smoothed CC values ----
cc74_smoothed = 0  # Filter
cc91_smoothed = 0  # Reverb
pitch_smoothed = 0  # Pitch Bend
gross_smoothed = 0  # Gross Beat

# ---- Helper functions ----
def smooth(prev, new, alpha=0.3):
    return prev + alpha * (new - prev)

def is_fist(hand_landmarks):
    threshold = 0.05
    for tip_id in [8, 12, 16, 20]:
        tip = hand_landmarks.landmark[tip_id]
        mcp = hand_landmarks.landmark[tip_id - 3]
        if math.hypot(tip.x - mcp.x, tip.y - mcp.y) > threshold:
            return False
    return True

def count_extended_fingers(hand_landmarks):
    count = 0
    for tip_id in [4, 8, 12, 16, 20]:
        tip = hand_landmarks.landmark[tip_id]
        mcp = hand_landmarks.landmark[tip_id - 3] if tip_id != 4 else hand_landmarks.landmark[2]
        if math.hypot(tip.x - mcp.x, tip.y - mcp.y) > 0.05:
            count += 1
    return count

# ---- Main loop ----
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

GROSS_BEAT_CC = 22  # assign to Gross Beat mix knob in FL

with mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_hand, right_hand = None, None
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == 'Right':
                    right_hand = hand_landmarks
                else:
                    left_hand = hand_landmarks

        # ---- RIGHT HAND: notes + filter ----
        if right_hand:
            if is_fist(right_hand):
                # Stop all notes
                for n in finger_notes.values():
                    send_note_off(n)
                for k in finger_states.keys():
                    finger_states[k] = False
            else:
                # Y-position → CC74 (filter)
                y_norm = 1.0 - right_hand.landmark[0].y
                cc74_new = int(y_norm * 127)
                cc74_smoothed = smooth(cc74_smoothed, cc74_new)
                outport.send(mido.Message('control_change', control=74, value=int(cc74_smoothed)))

                # Individual finger notes
                for tip_id, note in finger_notes.items():
                    tip = right_hand.landmark[tip_id]
                    pip = right_hand.landmark[tip_id - 2]
                    mcp = right_hand.landmark[tip_id - 3]
                    tip_mcp = math.hypot(tip.x - mcp.x, tip.y - mcp.y)
                    pip_mcp = math.hypot(pip.x - mcp.x, pip.y - mcp.y)
                    extended = tip_mcp > pip_mcp * 1.1

                    if extended and not finger_states[tip_id]:
                        velocity = int(min(127, 50 + (1.0 - tip.y) * 77))
                        send_note_on(note, velocity)
                        finger_states[tip_id] = True
                    elif not extended and finger_states[tip_id]:
                        send_note_off(note)
                        finger_states[tip_id] = False

        # ---- LEFT HAND: Gross Beat / Pitch Bend ----
        if left_hand:
            fingers_up = count_extended_fingers(left_hand)

            if fingers_up == 1:
                # One finger up → Gross Beat Mix control (use index finger Y)
                finger_y = left_hand.landmark[8].y
                gross_new = int((1.0 - finger_y) * 127)
                gross_smoothed = smooth(gross_smoothed, gross_new)
                outport.send(mido.Message('control_change', control=GROSS_BEAT_CC, value=int(gross_smoothed)))

            elif fingers_up >= 5:
                # All fingers open → Pitch bend
                x_norm = left_hand.landmark[0].x
                pitch_new = int((x_norm * 2 - 1) * 8191)
                pitch_smoothed = smooth(pitch_smoothed, pitch_new)
                outport.send(mido.Message('pitchwheel', pitch=int(pitch_smoothed)))

        # ---- BOTH HANDS: reverb based on distance ----
        if right_hand and left_hand:
            r = right_hand.landmark[0]
            l = left_hand.landmark[0]
            dist = math.hypot(r.x - l.x, r.y - l.y)
            cc91_new = int(min(127, dist * 90))
            cc91_smoothed = smooth(cc91_smoothed, cc91_new)
            outport.send(mido.Message('control_change', control=91, value=int(cc91_smoothed)))

        # ---- Draw and HUD ----
        if right_hand:
            mp_drawing.draw_landmarks(frame, right_hand, mp_hands.HAND_CONNECTIONS)
        if left_hand:
            mp_drawing.draw_landmarks(frame, left_hand, mp_hands.HAND_CONNECTIONS)

        y = 20
        cv2.putText(frame, f"Filter CC74: {int(cc74_smoothed)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,100,0), 1)
        y += 20
        cv2.putText(frame, f"Gross Beat Mix CC22: {int(gross_smoothed)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        y += 20
        cv2.putText(frame, f"Pitch Bend: {int(pitch_smoothed)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        y += 20
        cv2.putText(frame, f"Reverb CC91: {int(cc91_smoothed)}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,255), 1)

        cv2.imshow("Gesture Performance Mode", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
outport.close()
