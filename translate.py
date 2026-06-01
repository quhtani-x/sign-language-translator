import cv2
import mediapipe as mp

# SIGN LANGUAGE TRANSLATOR (finger-spelling helper).
# uses MediaPipe hand tracking to read which fingers you're holding up and maps
# common hand shapes to letters/words live from the webcam. it's a simple
# rule-based recognizer (counts extended fingers + thumb) - the foundation of an
# accessibility tool that turns hand signs into text.
# pip install opencv-python mediapipe

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# fingertip landmark ids and the joint below each, to tell if a finger is up
TIPS = [4, 8, 12, 16, 20]


def fingers_up(hand, handedness):
    # returns a list like [thumb, index, middle, ring, pinky] of 0/1
    lm = hand.landmark
    up = []
    # thumb: compare x (depends on left/right hand)
    if handedness == "Right":
        up.append(1 if lm[4].x < lm[3].x else 0)
    else:
        up.append(1 if lm[4].x > lm[3].x else 0)
    # other 4 fingers: tip higher than the joint below = up
    for tip in TIPS[1:]:
        up.append(1 if lm[tip].y < lm[tip - 2].y else 0)
    return up


def read_sign(up):
    # map simple finger patterns to a meaning
    total = sum(up)
    patterns = {
        (0, 0, 0, 0, 0): "A / fist",
        (1, 1, 1, 1, 1): "5 / hi",
        (0, 1, 0, 0, 0): "1 / D",
        (0, 1, 1, 0, 0): "2 / V",
        (0, 1, 1, 1, 0): "3 / W",
        (1, 1, 0, 0, 1): "I love you",
        (1, 0, 0, 0, 0): "thumbs up",
        (0, 0, 0, 0, 1): "pinky",
    }
    return patterns.get(tuple(up), f"{total} fingers")


def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror so it feels natural
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        text = "show a hand"
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            label = results.multi_handedness[0].classification[0].label
            up = fingers_up(hand, label)
            text = read_sign(up)
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.rectangle(frame, (0, 0), (640, 50), (30, 30, 30), -1)
        cv2.putText(frame, text, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 180), 2)
        cv2.imshow("sign language translator (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
