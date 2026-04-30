import os
import argparse
import cv2
from database.db import Database
from face.camera import Camera
from face.detector import FaceDetector
from face.recognizer import FaceRecognizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Name of the person")
    args = parser.parse_args()

    db = Database()
    camera = Camera()
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    
    camera.start()
    
    print("Press 's' to save the face, or 'q' to quit.")
    
    while True:
        frame = camera.read()
        if frame is None:
            continue
            
        rects = detector.detect(frame)
        
        display = frame.copy()
        if rects:
            largest = max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
            x1, y1, x2, y2 = largest.left(), largest.top(), largest.right(), largest.bottom()
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, "TARGET", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        cv2.imshow("Enroll Face", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and rects:
            emb = recognizer.compute_embedding(frame, largest)
            if emb is not None:
                pid = db.add_face(args.name, emb)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                qual = recognizer.calculate_quality(gray, largest)
                db.add_embedding(pid, emb, qual)
                print(f"Successfully saved {args.name} (ID: {pid})")
                break
            else:
                print("Failed to compute embedding.")
                
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
