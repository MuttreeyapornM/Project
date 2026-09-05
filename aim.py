"""Live aiming aid: shows whether the chessboard is fully detectable.

  ./venv/bin/python aim.py 4          # device index, default 4 (rear)
  ./venv/bin/python aim.py 4 13 4     # device, inner cols, inner rows

Green = full 13x4 grid found (a 14x5 board). Red = not found.
Yellow margin bars warn that the board is touching an image border, which is the
usual reason detection fails even when the board looks visible.
Press q to quit.
"""
import sys, cv2, numpy as np

dev = int(sys.argv[1]) if len(sys.argv) > 1 else 4
COLS = int(sys.argv[2]) if len(sys.argv) > 3 else 13
ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    sys.exit(f"cannot open /dev/video{dev}")
print(f"/dev/video{dev}: looking for {COLS}x{ROWS} inner corners "
      f"(a {COLS+1}x{ROWS+1} square board). q to quit.")

flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE
streak = 0
while True:
    ok, f = cap.read()
    if not ok:
        continue
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    found, cor = cv2.findChessboardCornersSB(g, (COLS, ROWS), flags)
    if not found:
        found, cor = cv2.findChessboardCornersSB(
            cv2.createCLAHE(2.0, (8, 8)).apply(g), (COLS, ROWS), flags)
    h, w = f.shape[:2]
    if found:
        streak += 1
        cv2.drawChessboardCorners(f, (COLS, ROWS), cor, True)
        p = cor.reshape(-1, 2)
        x0, x1, y0, y1 = p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max()
        m = min(x0, y0, w - x1, h - y1)
        msg = f"FOUND  {COLS}x{ROWS}   margin to edge: {m:.0f} px"
        col = (0, 200, 0) if m > 40 else (0, 200, 255)
        if m <= 40:
            msg += "  <-- TOO CLOSE TO EDGE, tilt away"
        cv2.rectangle(f, (int(x0), int(y0)), (int(x1), int(y1)), col, 3)
    else:
        streak = 0
        col = (0, 0, 255)
        msg = f"NOT FOUND - need the whole {COLS+1}x{ROWS+1} board in frame"
    cv2.rectangle(f, (0, 0), (w, 46), (0, 0, 0), -1)
    cv2.putText(f, msg, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.putText(f, f"stable frames: {streak}", (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow(f"aim /dev/video{dev}", cv2.resize(f, (w // 2, h // 2)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release(); cv2.destroyAllWindows()
