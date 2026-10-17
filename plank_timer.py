"""Détection de planche (plank) en direct avec chronomètre - MediaPipe Pose + OpenCV.

Usage :
    python plank_timer.py --list          # liste les caméras disponibles
    python plank_timer.py --camera 1      # lance la détection (touche q pour quitter)

Placez-vous de profil, en pied, à environ 2 m de la caméra.
Chaque planche (>= MIN_DURATION_S secondes) est enregistrée dans planche.db (SQLite).
"""
import argparse
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

# Indices des points MediaPipe Pose
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_ANKLE, R_ANKLE = 27, 28

# Réglages (à ajuster après vos essais)
MIN_ANGLE = 160        # angle épaule-hanche-cheville minimal (180 = corps droit)
MAX_SLOPE = 0.35       # pente max du corps (|dy| / |dx|) : corps presque horizontal
MIN_VISIBILITY = 0.5   # confiance minimale de MediaPipe sur les 3 points
GRACE_S = 0.7          # secondes sans détection tolérées avant d'arrêter le chrono
MIN_DURATION_S = 3.0   # durée minimale pour enregistrer une planche
DB_PATH = "planche.db"


def get_angle(a, b, c):
    """Angle en degrés au point b, entre les segments b->a et b->c."""
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def detect_plank(lm, width, height):
    """True si le corps est en position de planche (vue de profil).

    lm : liste de points MediaPipe (attributs x, y, visibility, coordonnées normalisées).
    On utilise le côté du corps le mieux visible, et des critères indépendants de la
    distance à la caméra (angle + pente du corps), contrairement à un seuil en pixels.
    """
    sides = ((L_SHOULDER, L_HIP, L_ANKLE), (R_SHOULDER, R_HIP, R_ANKLE))
    ids = max(sides, key=lambda s: sum(lm[i].visibility for i in s))
    shoulder, hip, ankle = (lm[i] for i in ids)
    if min(shoulder.visibility, hip.visibility, ankle.visibility) < MIN_VISIBILITY:
        return False

    ps, ph, pa = [(p.x * width, p.y * height) for p in (shoulder, hip, ankle)]
    angle = get_angle(ps, ph, pa)
    dx, dy = abs(pa[0] - ps[0]), abs(pa[1] - ps[1])
    horizontal = dx > 0 and dy / dx < MAX_SLOPE
    return angle >= MIN_ANGLE and horizontal


@dataclass
class PlankTimer:
    """Chronomètre qui tolère de brèves pertes de détection (clignotement MediaPipe)."""
    grace: float = GRACE_S
    min_duration: float = MIN_DURATION_S
    start: Optional[float] = None
    last_seen: float = 0.0

    def update(self, is_plank, now):
        """Retourne (en_cours, durée_courante, durée_terminée ou None)."""
        finished = None
        if is_plank:
            self.last_seen = now
            if self.start is None:
                self.start = now
        elif self.start is not None and now - self.last_seen > self.grace:
            finished = self._close()
        current = (now - self.start) if self.start is not None else 0.0
        return self.start is not None, current, finished

    def flush(self):
        """À appeler en fin de programme : clôt une planche encore en cours."""
        return self._close() if self.start is not None else None

    def _close(self):
        duration = self.last_seen - self.start
        self.start = None
        return duration if duration >= self.min_duration else None


def init_db(path=DB_PATH):
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE IF NOT EXISTS sessions ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "started_at TEXT NOT NULL, "
        "duration_s REAL NOT NULL)"
    )
    con.commit()
    return con


def save_session(con, duration, ended_at=None):
    ended_at = ended_at or datetime.now()
    started_at = ended_at - timedelta(seconds=duration)
    con.execute(
        "INSERT INTO sessions (started_at, duration_s) VALUES (?, ?)",
        (started_at.isoformat(timespec="seconds"), round(duration, 2)),
    )
    con.commit()


def open_camera(index):
    import cv2
    if sys.platform == "win32":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def list_cameras(max_index=5):
    """Teste les indices 0..max_index-1. Une caméra virtuelle (OBS) apparaît aussi."""
    for i in range(max_index):
        cap = open_camera(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        if ok:
            h, w = frame.shape[:2]
            print(f"Caméra {i} : image reçue ({w}x{h})")
        else:
            print(f"Caméra {i} : aucune image")
        cap.release()


def beep():
    try:
        import winsound
        winsound.Beep(1000, 200)
    except ImportError:
        pass  # pas de bip hors Windows


def main():
    parser = argparse.ArgumentParser(description="Détection de planche avec chronomètre")
    parser.add_argument("--camera", type=int, default=0, help="indice de la caméra")
    parser.add_argument("--list", action="store_true", help="liste les caméras et quitte")
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5,
                        min_tracking_confidence=0.5, model_complexity=1)

    cap = open_camera(args.camera)
    ok, _ = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        print(f"Impossible de lire la caméra {args.camera}. Essayez --list.")
        cap.release()
        return

    con = init_db()
    timer = PlankTimer()
    count = 0
    was_running = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Image non reçue, arrêt.")
                break

            h, w = frame.shape[:2]
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            is_plank = False
            if results.pose_landmarks:
                mp_draw.draw_landmarks(frame, results.pose_landmarks,
                                       mp_pose.POSE_CONNECTIONS)
                is_plank = detect_plank(results.pose_landmarks.landmark, w, h)

            running, current, finished = timer.update(is_plank, time.time())
            if running and not was_running:
                count += 1
                beep()  # un seul bip, au début de la planche
            was_running = running

            if finished is not None:
                save_session(con, finished)
                print(f"Planche {count} terminée : {finished:.1f} s (enregistrée)")

            text = f"Planche {count} : {current:.1f}s" if running else "Pas de planche"
            color = (0, 255, 0) if running else (0, 0, 255)
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Detection de planche (q pour quitter)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Interruption manuelle.")
    finally:
        last = timer.flush()
        if last is not None:
            save_session(con, last)
            print(f"Planche {count} terminée : {last:.1f} s (enregistrée)")
        cap.release()
        cv2.destroyAllWindows()
        con.close()


if __name__ == "__main__":
    main()