# SignSpeak

Train your own sign language on a webcam, then get live translations in a desktop UI.

It does **not** ship a giant pre-trained ASL dictionary. You teach it *your* signs (letters, words, or homemade gestures). MediaPipe tracks your hands; a small classifier maps those poses to the labels you captured.

## Run

Use Python 3.12 (MediaPipe is picky about newer versions):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Allow camera access if Windows asks.

## How to use

1. Open **Train**, type a meaning (`hello`, `a`, `thanks`).
2. Hold the pose in frame and press **Hold to capture**.
3. Repeat for at least two signs.
4. Click **Train model**.
5. Sign in front of the camera. Hold a pose for about a second to add it to the sentence.
6. **Speak** reads the sentence out loud.

Samples live in `data/` so you can close the app and keep training later.
