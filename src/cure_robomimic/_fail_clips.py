"""Collect FAILURE clips for BC / SafeDAgger / ELCD on Can, each from a
DIFFERENT seed. Writes both mp4 and gif."""
import sys, glob, numpy as np, imageio, cv2
from PIL import Image
import robomimic.utils.file_utils as FileUtils
from cure_robomimic.demo_bank import load_demo_bank
from cure_robomimic.env_utils import build_env
from cure_robomimic.policies import BCPolicy, DemoManifoldRecoveryEEF, SafeDAggerEEF, SwitchingPolicy
from cure_robomimic.rollout import RobomimicEvalConfig, rollout, make_task_detector
from cure_robomimic.tasks import TASKS

TASK, MAG = "can", 0.7
t = TASKS[TASK]
ck = sorted(glob.glob(f"outputs_robomimic/models/bc_{TASK}/*/*/models/model_epoch_600.pth"))[-1]
bank = load_demo_bank(t.dataset, cube_weight=1.0, point_stride=3)
pol, _ = FileUtils.policy_from_checkpoint(ckpt_path=ck, device="cuda", verbose=False)
bc = BCPolicy(rollout_policy=pol, name="BC")
env = build_env(t.dataset, offscreen=True, cam_h=256, cam_w=256)
cfg = RobomimicEvalConfig(perturb_step=t.perturb_step, push_steps=t.push_steps,
                          horizon=t.horizon, recover_window=t.recover_window)
det = make_task_detector(bank, t, env, bc, cfg)
methods = {
    "bc": (bc, "BC"),
    "safedagger": (SafeDAggerEEF(nominal=bc, bank=bank, detector=det,
                   object_pos_key=t.object_pos_key, name="SafeDAgger"), "SafeDAgger"),
    "elcd": (SwitchingPolicy(nominal=bc, recovery=DemoManifoldRecoveryEEF(bank=bank, object_pos_key=t.object_pos_key),
             detector=det, object_pos_key=t.object_pos_key, name="ELCD"), "ELCD"),
}


def label(frames, text):
    out = []
    for fr in frames:
        fr = np.ascontiguousarray(fr)
        cv2.rectangle(fr, (0, 0), (fr.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(fr, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1, cv2.LINE_AA)
        out.append(fr)
    return out


def save_gif(frames, path):
    imgs = [Image.fromarray(f[::1]).resize((160, 160)).convert("P", palette=Image.ADAPTIVE, colors=64)
            for f in frames[::3]]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=150, loop=0, optimize=True)


used = set()
for key, (pol_, disp) in methods.items():
    chosen = None
    for rid in range(60):
        if rid in used:
            continue
        r = rollout(env, pol_, bank, cfg, perturbation=MAG, seed=0, rollout_id=rid, record_frames=True)
        if not r.task_success:
            chosen = (rid, r); used.add(rid); break
    if chosen is None:
        print(f"{disp}: no unused failure seed found"); continue
    rid, r = chosen
    frames = label(r.frames, f"{disp}  -  FAILURE  (seed {rid})")
    base = f"outputs_robomimic/can/videos/canfail_{key}_seed{rid}"
    imageio.mimsave(base + ".mp4", frames, fps=20)
    save_gif(frames, base + ".gif")
    print(f"wrote {base}.mp4 + .gif  (seed={rid}, success={r.task_success})")
env.close()
