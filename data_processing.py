# import pickle
# import numpy as np

# # Load the pickle file
# with open("/home/wangyenjen/minbc/data/screw_driver_modified/train/0901_154711/2025-09-01T15-47-11-983274.pkl", "rb") as f:
#     data = pickle.load(f)

# # Check if it's a dict
# if isinstance(data, dict):
#     for key, value in data.items():
#         # Get type
#         vtype = type(value)

#         # If it's a NumPy array, show shape
#         if isinstance(value, np.ndarray):
#             print(f"{key}: {vtype}, shape = {value.shape}, dtype = {value.dtype}")
#         # If it's a list, show length
#         elif isinstance(value, list):
#             print(f"{key}: {vtype}, length = {len(value)}")
#         # If it's a dict, show sub-keys
#         elif isinstance(value, dict):
#             print(f"{key}: {vtype}, sub-keys = {list(value.keys())}")
#         else:
#             # For scalars, strings, or custom objects
#             print(f"{key}: {vtype}, value = {value}")
# else:
#     print(f"Top-level data is a {type(data)}")

import os
import pickle
import numpy as np

SRC_DIR = "/home/wangyenjen/screw_driver"
DST_DIR = "/home/wangyenjen/screw_driver_modified"

START_TOL = float(os.getenv("START_TOL", "1e-3"))  
END_TOL   = float(os.getenv("END_TOL",   "1e-3")) 
MIN_KEEP  = int(os.getenv("MIN_KEEP", "5"))

REMOVE_KEYS = ["gripper_position", "touch", "activated"]

bad_files = []
processed = skipped_missing_keys = skipped_shape_mismatch = 0
skipped_unreadable = skipped_all_trimmed = 0

def safe_load_pickle(path):
    try:
        if os.path.getsize(path) == 0:
            return None, "empty file"
    except OSError as e:
        return None, f"os error: {e}"
    try:
        with open(path, "rb") as f:
            return pickle.load(f), None
    except EOFError:
        return None, "EOFError (likely truncated)"
    except pickle.UnpicklingError:
        try:
            with open(path, "rb") as f:
                return pickle.load(f, encoding="latin1"), None
        except Exception as e2:
            return None, f"UnpicklingError: {e2}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def _as_np(x): return np.asarray(x)

def _vec_for_head(ctrl):
    a = _as_np(ctrl)
    a = a[0] if a.ndim >= 2 else a
    return a.reshape(-1)

def _vec_for_tail(arr):
    a = _as_np(arr)
    a = a[-1] if a.ndim >= 2 else a
    return a.reshape(-1)

def _compute_folder_trim(files):
    first_data, err = safe_load_pickle(files[0])
    if err: return 0, 0 
    last_data, err2 = safe_load_pickle(files[-1])
    if err2: return 0, 0

    if "control" not in first_data or "control" not in last_data or "xhand_act" not in last_data:
        return 0, 0

    ref_head_c = _vec_for_head(first_data["control"])
    ref_tail_c = _vec_for_tail(last_data["control"])
    ref_tail_x = _vec_for_tail(last_data["xhand_act"])

    dev_head = []  
    dev_tail_ok = []  
    for p in files:
        d, e = safe_load_pickle(p)
        if e or "control" not in d or "xhand_act" not in d:
            dev_head.append(np.inf)
            dev_tail_ok.append(True) 
            continue
        c_h = _vec_for_head(d["control"])
        c_t = _vec_for_tail(d["control"])
        x_t = _vec_for_tail(d["xhand_act"])
        dev_head.append(np.linalg.norm(c_h - ref_head_c))
        non_quiet = (np.linalg.norm(c_t - ref_tail_c) > END_TOL) or (np.linalg.norm(x_t - ref_tail_x) > END_TOL)
        dev_tail_ok.append(non_quiet)

    idxs = np.where(np.asarray(dev_head) > START_TOL)[0]
    start_idx = int(idxs[0]) if idxs.size else len(files)

    nonq = np.where(np.asarray(dev_tail_ok))[0]
    end_idx = int(nonq[-1] + 1) if nonq.size else 1

    if start_idx >= end_idx: return 0, 0
    if end_idx - start_idx < MIN_KEEP: return 0, 0
    return start_idx, end_idx

def _write_one(src_path, dst_path):
    global processed, skipped_missing_keys, skipped_shape_mismatch, skipped_unreadable

    data, err = safe_load_pickle(src_path)
    if err:
        print(f"[SKIP] {src_path} unreadable -> {err}")
        bad_files.append((src_path, err)); skipped_unreadable += 1; return

    if not isinstance(data, dict):
        print(f"[SKIP] {src_path} (non-dict {type(data)})")
        bad_files.append((src_path, f"non-dict: {type(data)}")); skipped_unreadable += 1; return

    if "control" not in data or "xhand_act" not in data:
        print(f"[SKIP] {src_path} (missing 'control' or 'xhand_act')"); skipped_missing_keys += 1; return

    control = _as_np(data["control"])
    xhand_act = _as_np(data["xhand_act"])

    if control.ndim >= 2 or xhand_act.ndim >= 2:
        Tc = control.shape[0] if control.ndim >= 2 else 1
        Tx = xhand_act.shape[0] if xhand_act.ndim >= 2 else 1
        if Tc != Tx:
            print(f"[SKIP] {src_path} time length mismatch: control T={Tc}, xhand_act T={Tx}")
            skipped_shape_mismatch += 1; return
        c2 = control.reshape(Tc, -1)
        x2 = xhand_act.reshape(Tx, -1)
        action = np.concatenate([c2, x2], axis=-1)
    else:
        action = np.concatenate([control.reshape(-1), xhand_act.reshape(-1)], axis=0)

    data["action"] = action

    if "xhand_tactile" in data:
        xt = _as_np(data["xhand_tactile"])
        if action.ndim == 2 and xt.ndim >= 1 and xt.shape[0] == action.shape[0]:
            data["xhand_tactile"] = xt.reshape(xt.shape[0], -1)
        else:
            data["xhand_tactile"] = xt.reshape(-1)

    for k in REMOVE_KEYS + ["control", "xhand_act"]:
        data.pop(k, None)

    parent = os.path.dirname(dst_path) or DST_DIR
    os.makedirs(parent, exist_ok=True)
    tmp = dst_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, dst_path)
    processed += 1
    # print(f"[OK] Saved: {dst_path}")

def process_folder(root, files):
    global skipped_all_trimmed
    files = sorted(files)
    if not files: return
    start, end = _compute_folder_trim(files)
    if start >= end:
        print(f"[SKIP][FOLDER] {root} trimmed to empty or < MIN_KEEP")
        skipped_all_trimmed += 1
        return
    keep = files[start:end]
    print(f"[FOLDER] {root} keep {len(keep)}/{len(files)} (range {start}:{end})")
    for src_path in keep:
        rel_path = os.path.relpath(src_path, SRC_DIR)
        if rel_path.startswith(os.pardir):
            print(f"[SKIP] {src_path} unsafe relpath"); continue
        dst_path = os.path.join(DST_DIR, rel_path)
        _write_one(src_path, dst_path)

def main():
    os.makedirs(DST_DIR, exist_ok=True)

    folder_to_files = {}
    for root, _, files in os.walk(SRC_DIR):
        pkl_paths = [os.path.join(root, f) for f in files if f.endswith(".pkl")]
        if pkl_paths:
            folder_to_files[root] = pkl_paths

    for root, files in sorted(folder_to_files.items()):
        process_folder(root, files)

    print("\n=== Summary ===")
    print(f"Processed OK: {processed}")
    print(f"Skipped (unreadable): {skipped_unreadable}")
    print(f"Skipped (missing keys): {skipped_missing_keys}")
    print(f"Skipped (shape mismatch): {skipped_shape_mismatch}")
    print(f"Skipped (all-trimmed/too short folders): {skipped_all_trimmed}")
    if bad_files:
        print("\nUnreadable files (first 20):")
        for p, e in bad_files[:20]: print(f" - {p} -> {e}")
        if len(bad_files) > 20: print(f" ... and {len(bad_files) - 20} more")

if __name__ == "__main__":
    main()

