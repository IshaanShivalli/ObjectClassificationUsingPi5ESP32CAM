#!/usr/bin/env python3
"""Pi 5: pull JPEG snapshots from the ESP32-CAM over WiFi and classify with an Edge Impulse .eim.

Install: pip install requests opencv-python-headless numpy edge_impulse_linux
Run:     python live_classify_wifi.py --model model/modelfile.eim --cam http://<cam-ip>
"""
import argparse
import time

import cv2
import numpy as np
import requests
from edge_impulse_linux.image import ImageImpulseRunner


def get_features(runner, rgb):
    try:
        feats, cropped = runner.get_features_from_image_auto_studio_settings(rgb)
    except AttributeError:
        feats, cropped = runner.get_features_from_image(rgb)
    return feats, cropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cam", required=True, help="e.g. http://192.168.1.50")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    url = args.cam.rstrip("/") + "/capture"
    sess = requests.Session()

    with ImageImpulseRunner(args.model) as runner:
        print("Model:", runner.init()["project"]["name"])
        last = time.time()
	while True:
            try:
                r = sess.get(url, timeout=5)
                r.raise_for_status()
            except requests.RequestException as e:
                print("fetch failed:", e)
                time.sleep(1)
                continue

            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feats, _ = get_features(runner, rgb)
            res = runner.classify(feats)["result"]

            boxes = [b for b in res.get("bounding_boxes", []) if b["value"] >= args.min_conf]
            now = time.time()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            print(f"{fps:4.1f} fps | " + (", ".join(f"{b['label']} {b['value']:.2f}" for b in boxes) or "none"))

            if args.show:
                for b in boxes:
                    x, y, w, h = b["x"], b["y"], b["width"], b["height"]
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, f"{b['label']} {b['value']:.2f}", (x, max(y - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("ESP32-CAM", img)
                if cv2.waitKey(1) == 27:
                    break


if __name__ == "__main__":
    main()
