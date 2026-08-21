import json
import math
import os
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = {
    "model": "yolo11n.pt",
    "confidence": 0.35,
    "iou": 0.5,
    "process_every_n_frames": 2,
    "unattended_seconds": 8,
    "item_person_max_distance_px": 180,
    "item_stationary_movement_px": 35,
    "zones": {
        "entry": [0.00, 0.00, 0.30, 1.00],
        "reception": [0.30, 0.00, 0.70, 1.00],
        "doctor": [0.70, 0.00, 1.00, 1.00]
    }
}

PERSON_CLASS = 0
ITEM_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}


def load_config():
    p = APP_DIR / "config.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            merged = DEFAULT_CONFIG.copy()
            merged.update({k: v for k, v in data.items() if k != "zones"})
            merged["zones"] = DEFAULT_CONFIG["zones"].copy()
            merged["zones"].update(data.get("zones", {}))
            return merged
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_CONFIG))


def save_config(config):
    (APP_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def point_in_zone(pt, zone, width, height):
    x, y = pt
    x1, y1, x2, y2 = [float(v) for v in zone]
    return x1 * width <= x <= x2 * width and y1 * height <= y <= y2 * height


def zone_name(pt, zones, width, height):
    for name, zone in zones.items():
        if point_in_zone(pt, zone, width, height):
            return name
    return "outside"


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class PersonState:
    track_id: int
    first_seen: float
    first_entry: float | None = None
    reception_enter: float | None = None
    doctor_enter: float | None = None
    waiting_seconds: float | None = None
    last_zone: str = "outside"
    last_seen: float = 0.0
    reentry_count: int = 0


@dataclass
class ItemState:
    track_id: int
    label: str
    first_seen: float
    last_seen: float
    first_person_id: int | None = None
    last_person_id: int | None = None
    last_person_distance: float | None = None
    stationary_since: float | None = None
    max_stationary_time: float = 0.0
    alerted: bool = False
    last_center: tuple[float, float] | None = None


def process_video(video_path, config, model, frame_placeholder, metrics_placeholder, event_callback=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    process_n = max(1, int(config["process_every_n_frames"]))

    persons: dict[int, PersonState] = {}
    items: dict[int, ItemState] = {}
    events = []
    frame_index = 0
    last_processed = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            video_seconds = frame_index / fps

            if frame_index % process_n != 0:
                if last_processed is not None:
                    frame_placeholder.image(last_processed, channels="BGR", use_container_width=True)
                continue

            result = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=float(config["confidence"]),
                iou=float(config["iou"]),
                tracker="bytetrack.yaml",
                classes=[PERSON_CLASS, *ITEM_CLASSES.keys()],
                verbose=False,
            )[0]

            person_points = {}
            item_points = {}
            annotated = frame.copy()

            # Draw zones.
            zone_labels = {"entry": "ENTRY", "reception": "RECEPTION", "doctor": "DOCTOR / SERVICE"}
            for zname, z in config["zones"].items():
                x1, y1, x2, y2 = [int(float(v) * s) for v, s in zip(z, [width, height, width, height])]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(annotated, zone_labels.get(zname, zname.upper()), (x1 + 5, max(22, y1 + 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                cls = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                xyxy = boxes.xyxy.cpu().numpy()
                names = result.names

                for track_id, cls_id, conf, box in zip(ids, cls, confs, xyxy):
                    box = [float(v) for v in box]
                    pt = center(box)
                    if cls_id == PERSON_CLASS:
                        person_points[int(track_id)] = pt
                        zname = zone_name(pt, config["zones"], width, height)
                        state = persons.get(int(track_id))
                        if state is None:
                            state = PersonState(track_id=int(track_id), first_seen=video_seconds, last_seen=video_seconds)
                            persons[int(track_id)] = state

                        previous_zone = state.last_zone
                        state.last_seen = video_seconds
                        if zname != previous_zone and zname != "outside" and previous_zone != "outside":
                            state.reentry_count += 1
                        state.last_zone = zname

                        if zname == "entry" and state.first_entry is None:
                            state.first_entry = video_seconds
                        if zname == "reception" and state.reception_enter is None:
                            state.reception_enter = video_seconds
                        if zname == "doctor" and state.doctor_enter is None:
                            state.doctor_enter = video_seconds
                            if state.reception_enter is not None and state.waiting_seconds is None:
                                state.waiting_seconds = max(0.0, state.doctor_enter - state.reception_enter)
                                events.append({
                                    "timestamp_s": round(video_seconds, 2),
                                    "event": "WAITING_TIME_COMPLETED",
                                    "person_id": int(track_id),
                                    "waiting_seconds": round(state.waiting_seconds, 2),
                                    "waiting_minutes": round(state.waiting_seconds / 60, 2),
                                })
                                if event_callback:
                                    event_callback(events[-1])

                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 220, 50), 2)
                        label = f"Person {track_id} | {zname}"
                        cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (50, 220, 50), 2, cv2.LINE_AA)

                    elif cls_id in ITEM_CLASSES:
                        item_points[int(track_id)] = (pt, ITEM_CLASSES[cls_id])
                        state = items.get(int(track_id))
                        if state is None:
                            state = ItemState(track_id=int(track_id), label=ITEM_CLASSES[cls_id], first_seen=video_seconds,
                                              last_seen=video_seconds, last_center=pt)
                            items[int(track_id)] = state
                            if person_points:
                                pid, dist = min(((pid, distance(pt, p)) for pid, p in person_points.items()), key=lambda x: x[1])
                                if dist <= float(config["item_person_max_distance_px"]):
                                    state.first_person_id = pid
                                    state.last_person_id = pid
                                    state.last_person_distance = dist

                        state.last_seen = video_seconds
                        movement = distance(pt, state.last_center) if state.last_center else 0.0
                        if movement <= float(config["item_stationary_movement_px"]):
                            if state.stationary_since is None:
                                state.stationary_since = video_seconds
                            state.max_stationary_time = video_seconds - state.stationary_since
                        else:
                            state.stationary_since = None
                            state.max_stationary_time = 0.0
                        state.last_center = pt

                        if person_points:
                            pid, dist = min(((pid, distance(pt, p)) for pid, p in person_points.items()), key=lambda x: x[1])
                            state.last_person_id = pid
                            state.last_person_distance = dist

                        if (not state.alerted and state.stationary_since is not None and
                            state.max_stationary_time >= float(config["unattended_seconds"]) and
                            (state.last_person_distance is None or state.last_person_distance > float(config["item_person_max_distance_px"]))):
                            state.alerted = True
                            event = {
                                "timestamp_s": round(video_seconds, 2),
                                "event": "UNATTENDED_ITEM",
                                "item_id": int(track_id),
                                "item_type": state.label,
                                "last_person_id": state.last_person_id,
                                "stationary_seconds": round(state.max_stationary_time, 2),
                                "person_distance_px": round(state.last_person_distance, 1) if state.last_person_distance is not None else None,
                            }
                            events.append(event)
                            if event_callback:
                                event_callback(event)

                        x1, y1, x2, y2 = map(int, box)
                        line_color = (0, 0, 255) if state.alerted else (255, 180, 0)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), line_color, 2)
                        txt = f"{state.label} {track_id}" + (" | UNATTENDED" if state.alerted else "")
                        cv2.putText(annotated, txt, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    line_color, 2, cv2.LINE_AA)

            # Overlay summary.
            completed = [p for p in persons.values() if p.waiting_seconds is not None]
            unattended = [i for i in items.values() if i.alerted]
            avg_wait = (sum(p.waiting_seconds for p in completed) / len(completed)) if completed else 0
            cv2.rectangle(annotated, (10, 10), (370, 105), (20, 20, 20), -1)
            cv2.putText(annotated, f"People tracked: {len(persons)}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240,240,240), 2)
            cv2.putText(annotated, f"Completed waits: {len(completed)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240,240,240), 2)
            cv2.putText(annotated, f"Avg wait: {avg_wait/60:.1f} min | Alerts: {len(unattended)}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 2)

            last_processed = annotated.copy()
            frame_placeholder.image(annotated, channels="BGR", use_container_width=True)
            progress = frame_index / total_frames if total_frames else 0
            metrics_placeholder.progress(min(1.0, progress), text=f"Processing {frame_index}/{total_frames or '?'} frames")
    finally:
        cap.release()

    person_rows = []
    for p in persons.values():
        person_rows.append({
            "person_id": p.track_id,
            "entry_time_s": round(p.first_entry, 2) if p.first_entry is not None else None,
            "reception_time_s": round(p.reception_enter, 2) if p.reception_enter is not None else None,
            "doctor_time_s": round(p.doctor_enter, 2) if p.doctor_enter is not None else None,
            "waiting_seconds": round(p.waiting_seconds, 2) if p.waiting_seconds is not None else None,
            "waiting_minutes": round(p.waiting_seconds / 60, 2) if p.waiting_seconds is not None else None,
            "last_zone": p.last_zone,
            "reentries": p.reentry_count,
        })

    item_rows = []
    for i in items.values():
        item_rows.append({
            "item_id": i.track_id,
            "item_type": i.label,
            "last_person_id": i.last_person_id,
            "stationary_seconds": round(i.max_stationary_time, 2),
            "last_person_distance_px": round(i.last_person_distance, 1) if i.last_person_distance is not None else None,
            "unattended_alert": i.alerted,
        })

    return {
        "person_df": pd.DataFrame(person_rows),
        "item_df": pd.DataFrame(item_rows),
        "events_df": pd.DataFrame(events),
        "video_meta": {"fps": fps, "frames": total_frames, "width": width, "height": height, "duration_s": total_frames / fps if fps else None},
    }


def main():
    st.set_page_config(page_title="Hospital AI CCTV POC", layout="wide")
    st.title("Hospital AI CCTV Video Analytics — POC")
    st.caption("Local/on-premise demonstration: person tracking, waiting-time analytics, unattended-item alerts, and last-person association.")

    config = load_config()
    with st.sidebar:
        st.header("POC Settings")
        config["model"] = st.selectbox("YOLO model", ["yolo11n.pt", "yolo11s.pt", "yolov8n.pt"], index=["yolo11n.pt", "yolo11s.pt", "yolov8n.pt"].index(config.get("model", "yolo11n.pt")))
        config["confidence"] = st.slider("Detection confidence", 0.15, 0.80, float(config["confidence"]), 0.05)
        config["iou"] = st.slider("Tracking IoU", 0.20, 0.80, float(config["iou"]), 0.05)
        config["process_every_n_frames"] = st.slider("Process every Nth frame", 1, 6, int(config["process_every_n_frames"]))
        config["unattended_seconds"] = st.slider("Unattended duration (sec)", 3, 60, int(config["unattended_seconds"]))
        config["item_person_max_distance_px"] = st.slider("Person-item separation (px)", 80, 500, int(config["item_person_max_distance_px"]), 10)
        config["item_stationary_movement_px"] = st.slider("Stationary movement threshold (px)", 10, 100, int(config["item_stationary_movement_px"]), 5)
        st.divider()
        st.subheader("Zones (normalized 0–1)")
        for zname in ["entry", "reception", "doctor"]:
            st.markdown(f"**{zname.title()}**")
            z = config["zones"][zname]
            c1, c2 = st.columns(2)
            z[0] = c1.number_input(f"{zname} x1", 0.0, 1.0, float(z[0]), 0.01, key=f"{zname}x1")
            z[1] = c2.number_input(f"{zname} y1", 0.0, 1.0, float(z[1]), 0.01, key=f"{zname}y1")
            c3, c4 = st.columns(2)
            z[2] = c3.number_input(f"{zname} x2", 0.0, 1.0, float(z[2]), 0.01, key=f"{zname}x2")
            z[3] = c4.number_input(f"{zname} y2", 0.0, 1.0, float(z[3]), 0.01, key=f"{zname}y2")
        if st.button("Save settings"):
            save_config(config)
            st.success("Saved to config.json")

    uploaded = st.file_uploader("Upload CCTV footage", type=["mp4", "avi", "mov", "mkv"])
    if not uploaded:
        st.info("Upload a hospital reception CCTV video to start the POC.")
        st.markdown("### POC flow")
        st.markdown("**Person detection → tracking → reception waiting time → doctor/service stage → unattended item → last associated person**")
        return

    if YOLO is None:
        st.error("Ultralytics is not installed. Run: pip install -r requirements.txt")
        return

    temp_dir = Path(tempfile.gettempdir()) / "hospital_cctv_poc"
    temp_dir.mkdir(parents=True, exist_ok=True)
    video_path = temp_dir / uploaded.name
    video_path.write_bytes(uploaded.getbuffer())

    if "poc_model_name" not in st.session_state or st.session_state["poc_model_name"] != config["model"]:
        with st.spinner(f"Loading {config['model']} (first run may download model weights)..."):
            st.session_state["poc_model"] = YOLO(config["model"])
            st.session_state["poc_model_name"] = config["model"]
    model = st.session_state["poc_model"]

    if st.button("Run POC analysis", type="primary"):
        st.session_state.pop("poc_result", None)
        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()
        started = time.time()
        with st.spinner("Running AI analysis..."):
            result = process_video(video_path, config, model, frame_placeholder, metrics_placeholder)
        st.session_state["poc_result"] = result
        st.session_state["poc_runtime"] = time.time() - started
        metrics_placeholder.success(f"Analysis complete in {st.session_state['poc_runtime']:.1f}s")

    result = st.session_state.get("poc_result")
    if result:
        person_df = result["person_df"]
        item_df = result["item_df"]
        events_df = result["events_df"]
        completed = person_df.dropna(subset=["waiting_seconds"]) if not person_df.empty else person_df
        avg_wait = completed["waiting_minutes"].mean() if not completed.empty else 0
        alerts = int(item_df["unattended_alert"].sum()) if not item_df.empty and "unattended_alert" in item_df else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("People tracked", len(person_df))
        c2.metric("Completed waits", len(completed))
        c3.metric("Average wait", f"{avg_wait:.1f} min")
        c4.metric("Unattended alerts", alerts)

        t1, t2, t3 = st.tabs(["Waiting-time analytics", "Unattended items", "Event log"])
        with t1:
            st.dataframe(person_df, use_container_width=True, hide_index=True)
            csv = person_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download waiting-time CSV", csv, "waiting_times.csv", "text/csv")
        with t2:
            st.dataframe(item_df, use_container_width=True, hide_index=True)
            csv = item_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download item CSV", csv, "unattended_items.csv", "text/csv")
        with t3:
            st.dataframe(events_df, use_container_width=True, hide_index=True)
            csv = events_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download event CSV", csv, "events.csv", "text/csv")

        st.caption(f"Video: {uploaded.name} | {result['video_meta']['width']}×{result['video_meta']['height']} | {result['video_meta']['fps']:.1f} FPS | {result['video_meta']['duration_s']:.1f}s")


if __name__ == "__main__":
    main()
