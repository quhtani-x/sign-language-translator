# Sign Language Translator

Reads hand signs from your webcam and turns them into text, live. It uses
MediaPipe hand tracking to see which fingers you're holding up, then maps common
hand shapes to letters/words (fist, peace sign, "I love you", thumbs up, etc.).
The foundation of an accessibility tool that converts finger-spelling to text.

## how it works

- MediaPipe finds 21 hand landmarks
- works out which fingers are extended (tip above the knuckle)
- a rule table maps the finger pattern to a meaning
- handles left/right hand for the thumb

## run

```bash
pip install opencv-python mediapipe
python translate.py
```

hold your hand up to the camera and try a fist, peace sign, or 🤟.

tags: ai, computer-vision, mediapipe, accessibility, sign-language

would love to grow this into full ASL alphabet recognition with a trained model.
