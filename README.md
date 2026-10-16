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
3. Wrist camera: place the tag on the board and press Enter in the terminal. `--marker-size` must be the measured black-square side of the printed tag.
   - **Automated (default, `--wrist-mode auto`):** see below. It asks once before the first move; `q` in the window or Ctrl-C stops it, and it fits what was captured.
   - **Manual (`--wrist-mode manual`):** pose the arm in teaching mode and capture views in the window (~10: varied heights, tilts, wrist angles). Keys: `Enter` capture, `d` undo, `t` re-measure the tag, `q` fit and finish.

### Automated wrist step (`wrist_camera_auto.py`)

The arm must be enabled, **out of** teaching mode, with the workspace clear.

1. The top camera measures the tag's pose on the board.
2. Views are planned on a cap above the tag (`--view-tilt` 20/35/50° from straight down, `--view-dist` 250/330/410 mm, `--view-azimuths` 8 directions), solved by IK on the same kinematics as the fit. The IK keeps the tag center exactly at `--tag-pixel` (default 50% across, 35% down: above the gripper, ±`--pixel-jitter` px), and gets the camera as close to the planned position as the arm can reach. A view is kept only if:
   - it ends up within 15% of the distance range and 10° of the tilt range;
   - the whole tag is inside the image;
   - the joints stay `--limit-margin` 5° inside the limits, with |j4| and |j6| ≤ `--wrist-range` 90° for the camera cable;
   - every link, the gripper tip and the camera stay `--clearance` 40 mm above the board;
   - the tag stays out of `--gripper-mask`, the part of the wrist image where the gripper fingers show (image fractions x0 y0 x1 y1, default `0.36 0.68 0.64 1.0`: bottom center, gripper half open). The window draws it in yellow; adjust it if the fingers show elsewhere, or `0 0 0 0` to turn it off.
3. Views are visited most different first. Each move streams a target at 50 Hz that slides along the straight joint-space line, with a smooth start and stop and a peak of `--joint-speed` 20°/s on the joint that moves most. The target waits whenever the arm lags more than `--track-tol` 3°, so the arm stays on the line. (A single far target let the Piper take its own path, and that touched the board.) The line must stay `--transit-clearance` 80 mm above the board, easing to the views' own `--clearance` over the first and last quarter. A line that isn't clear goes through the start pose; if that isn't clear either, the view is skipped. `--speed` 50% is only the Piper's own cap.
4. After the arm settles, it captures the tag and joints with the same checks as the manual step. Every `--refit-after` 4 views the planning mount is refitted and the remaining views are re-solved.
5. From `--min-views` 8 on, it fits the mount and j1..j5 offsets after every view. It stops when the leave-one-out median is under `--target-loo-mm` 3.0 and the mount moved less than `--converge-mm` 0.5 / `--converge-deg` 0.1 since the previous fit, or at `--max-views` 20. A run that didn't converge isn't written to the static dir.
6. The arm goes back to its start pose.

Planning starts from a rough mount: `--mount-guess`, else the existing `wrist_calibration.json`, else `nominal_wrist_mount.json` (station 11's measured mount, the same mount design everywhere).

Commands go to `--command-topic`, default `/joint_states`: `start_station_calibration.launch.py` remaps the Piper's input there, outside the station namespace. Each command sets all six joints by name, holds the gripper at its current opening (`position[6]`) and sets the Piper's speed cap (`velocity[6]`).

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
| `wrist_camera_auto.py` | automated wrist step: view planning, IK, motion, stop rule |
| `nominal_wrist_mount.json` | starting mount guess for planning |
| `breadboard_model.py` | Piper forward kinematics shared by the fits |
| `joint_offset_fit.py` | Piper DH table (and an offline joint-offset fit) |
