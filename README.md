# multi_camera_calibration

Station camera calibration for the AgileX Piper stations: the fixed top RealSense camera from the holes of a Thorlabs MB2424 breadboard, then the wrist RealSense camera from one ArUco tag lying on the board. Split out of [handeye_calibration_ros](https://github.com/skim743/handeye_calibration_ros).

## Requirements

- ROS 2 Humble (rclpy, sensor_msgs), sourced.
- `piper_ros` with `start_station_calibration.launch.py`, plus the `realsense2_camera` and `v4l2_camera` packages it starts.
- Python 3.10+, numpy, scipy, OpenCV with the `aruco` module (`opencv-contrib-python`).
- An existing `/armlab_ws/static_data` folder (or `--static-dir`).

No colcon build: run the scripts with `python3` from this folder.

## Run

```
ros2 launch piper start_station_calibration.launch.py
python3 multi_camera_calibration.py                   # top, then wrist
python3 multi_camera_calibration.py --cameras wrist   # wrist only, with the saved top calibration
```

1. If `top_calibration.json` or `wrist_calibration.json` already exists in the static dir (only the ones being calibrated are checked), it asks to overwrite or cancel before anything starts.
2. Top camera: the board must be clear of everything, tag included. When the hole count and jitter settle, press `e`. A fit over `--max-extrinsic-rms` is refused (`r` resets, then `e` again). `q` aborts.
   The board's 5 counterbored mounting holes (11.5 mm: center, and 11 pitches diagonally out from it) are circled orange. When 2 or more are found, they decide which detected hole is which. This keeps the board placement right when rows of holes are missed, e.g. dark near the arm base. They are also used in the pose solve. With fewer than 2, the window is the 24×24 block holding the most holes, as before, and the status line says so.
3. Wrist camera: place the tag on the board, press Enter in the terminal, then pose the arm in teaching mode and capture views in the window (~10: varied heights, tilts, wrist angles). Keys: `Enter` capture, `d` undo, `t` re-measure the tag, `q` fit and finish. `--marker-size` must be the measured black-square side of the printed tag.

## Output

- `<static-dir>/top_calibration.json`: top camera pose in the robot base frame.
- `<static-dir>/wrist_calibration.json`: wrist camera pose in the flange (end_pose) frame, plus the j1..j5 zero offsets (fitted from 8 views on).

Both carry `timestamp`, `camera`, `station_id` and `history_file`. The timestamped copies and the raw samples go under `--data-save-path` (default `./data`). A wrist fit whose joint offsets hit the bound is not written to the static dir.

The station number comes from the hostname (`robotics-education-labN`), as in the launch file. Topics default to `/stationN/intel_realsense_d435i_{top,wrist}/color/...` and `/stationN/joint_states_single`, and `ROS_DOMAIN_ID` is set to N. Use `--station-id` off-station. Every topic can be overridden (`--top-topic`, `--wrist-topic`, `--joint-topic`, ...).

## Files

| File | Role |
|---|---|
| `multi_camera_calibration.py` | entry point: top step, then wrist step |
| `top_camera_calibration.py` | hole detection, grid fit, top camera extrinsic; also runs standalone |
| `wrist_camera_calibration.py` | tag capture and wrist camera fit; also runs standalone (`run`, `fit`) |
| `breadboard_model.py` | Piper forward kinematics shared by the fits |
| `joint_offset_fit.py` | Piper DH table (and an offline joint-offset fit) |
