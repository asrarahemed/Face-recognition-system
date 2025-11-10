# deepface_stylish_realtime.py
import cv2
import time
from deepface import DeepFace

def draw_label(img, text, x, y, font_scale=0.6, thickness=1, padding=6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x - padding, y - padding - h), (x + w + padding, y + padding), (0,0,0), -1)  # filled background
    cv2.putText(img, text, (x, y), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)

def main(camera_index=0, resize_factor=0.5):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    fps_time = time.time()
    last_analysis = None
    analysis_interval = 0.8  # seconds between DeepFace analyze calls (lower -> more CPU)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # create smaller copy for faster analysis
            small = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
            h0, w0 = frame.shape[:2]
            h1, w1 = small.shape[:2]
            fx = w0 / w1
            fy = h0 / h1

            # Run DeepFace analyze periodically instead of every frame for speed
            now = time.time()
            if (last_analysis is None) or (now - last_analysis > analysis_interval):
                try:
                    # enforce_detection=False reduces failures when a face is partially visible
                    analysis = DeepFace.analyze(img_path = small, actions = ['age','gender','emotion'], enforce_detection=False)
                    last_analysis = now
                    last_result = analysis
                except Exception as e:
                    # keep last_result if analyse fails momentarily
                    last_result = None
                    # print("DeepFace error:", e)

            # If result found, render stylish overlay
            if 'last_result' in locals() and last_result:
                res = last_result
                # DeepFace may return a list of faces or a dict for single face
                if isinstance(res, list) and len(res) > 0:
                    res0 = res[0]
                else:
                    res0 = res

                # region coordinates on the small image
                region = res0.get('region', None)
                if region:
                    x = int(region['x'] * fx)
                    y = int(region['y'] * fy)
                    w = int(region['w'] * fx)
                    h = int(region['h'] * fy)

                    # Draw rounded-ish rectangle (simple rectangle here)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (30, 144, 255), 2)

                    # Prepare labels
                    dominant_emotion = res0.get('dominant_emotion', '')
                    age = res0.get('age', '')
                    gender = res0.get('gender', '')
                    label1 = f"{gender} | {age}"
                    label2 = f"{dominant_emotion}"

                    # Draw small stylish panels
                    draw_label(frame, label1, x, y - 10)
                    draw_label(frame, label2, x, y + h + 25)

            # Add FPS / instructions footer
            curr_fps = 1.0 / max(1e-6, time.time() - fps_time)
            fps_time = time.time()
            footer_text = f"DeepFace (no-train) - Press 'q' to quit | FPS: {curr_fps:.1f}"
            draw_label(frame, footer_text, 10, frame.shape[0] - 10, font_scale=0.5, thickness=1, padding=8)

            # Show window
            cv2.imshow("Stylish DeepFace Realtime", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
