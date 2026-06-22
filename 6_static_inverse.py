# -*- coding: utf-8 -*-
# Per-frame INDEPENDENT STATIC inverse (Jython). Run via Scripts -> Run Script.
# Uses ONLY Java I/O (ArtiSynth Jython has no Python stdlib).
# Each frame: reset -> pin targets -> settle -> record activations. Independent.
from maspack.matrix import Point3d
TARGETS_CSV = r"C:\Users\d11\Project\Tongue_Inverse\mri_fit\frame_targets_m.csv"
OUT_CSV     = r"C:\Users\d11\Project\Tongue_Inverse\mri_fit\activations_static_per_frame.csv"
CONTROLLER  = "mriTracking"
SETTLE_T    = 0.4
FPS         = 5.0
def get_controller():
    ctrls = root().getControllers()
    try:
        c = ctrls.get(CONTROLLER)
        if c is not None:
            return c
    except:
        pass
    for c in ctrls:
        if c.getClass().getSimpleName() == "TrackingController":
            return c
    return None
def get_tongue():
    mech = root().models().get(0)
    return mech.models().get(0)
def load_targets(path):
    from java.io import BufferedReader, FileReader
    frames = {}
    rd = BufferedReader(FileReader(path))
    rd.readLine()
    while True:
        line = rd.readLine()
        if line is None:
            break
        line = line.strip()
        if line == "":
            continue
        p = line.split(",")
        fr = int(p[0]); idx = int(p[1])
        frames.setdefault(fr, {})[idx] = (float(p[2]), float(p[3]), float(p[4]))
    rd.close()
    return frames
def deactivate_input_probes():
    for ip in root().getInputProbes():
        try:
            ip.setActive(False)
        except:
            ip.setActive(0)
def run_static_inverse():
    from java.io import File, FileWriter, PrintWriter
    ctrl = get_controller()
    if ctrl is None:
        print "ERROR: controller '%s' not found. Load model with manifest first." % CONTROLLER
        return
    tpts = ctrl.getTargetPoints()
    tongue = get_tongue()
    exciters = list(tongue.getMuscleExciters())
    names = [ex.getName() for ex in exciters]
    print "Controller OK: %d target points, %d exciters" % (tpts.size(), len(exciters))
    frames = load_targets(TARGETS_CSV)
    frame_ids = sorted(frames.keys())
    print "Loaded %d frames" % len(frame_ids)
    records = []
    n = min(tpts.size(), 11)
    for fr in frame_ids:
        reset()
        deactivate_input_probes()
        addBreakPoint(SETTLE_T)
        pts = frames[fr]
        for i in range(n):
            if i in pts:
                x, y, z = pts[i]
                tpts.get(i).setPosition(Point3d(x, y, z))
        run()
        waitForStop()
        acts = {}
        for ex in exciters:
            acts[ex.getName()] = ex.getExcitation()
        records.append((fr, acts))
        top = sorted(acts.items(), key=lambda kv: -kv[1])[:3]
        print "frame %3d -> " % fr + ", ".join("%s=%.3f" % (k, v) for k, v in top)
    fo = PrintWriter(FileWriter(File(OUT_CSV)))
    fo.println("frame,time," + ",".join(names))
    for fr, acts in records:
        t = (fr - 1) / FPS
        fo.println("%d,%.4f," % (fr, t) + ",".join("%.5f" % acts.get(nm, 0.0) for nm in names))
    fo.close()
    reset()
    print "DONE. wrote %d frames -> %s" % (len(records), OUT_CSV)
run_static_inverse()
