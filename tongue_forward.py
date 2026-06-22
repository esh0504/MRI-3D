#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tongue_forward.py  --  Python client for the ArtiSynth forward muscle model.

Start the server first inside ArtiSynth (Scripts -> Run Script -> forward_server.py),
then from Python:

    from tongue_forward import muscle_power, muscle_names, save_obj
    names = muscle_names()                  # exciter order (GGP, GGA, ...)
    verts, faces = muscle_power([0.1,0.2,0.3, ...])   # activations 0..1, in `names` order
    # verts: (Nverts,3) deformed tongue-surface nodes (model metres); faces: (Nfaces,3)
    save_obj(verts, faces, "pose.obj")

You can also pass a dict: muscle_power({"GGP":0.3, "HG":0.2}).
Each call: server resets to rest, applies the activations open-loop, settles to
equilibrium with the real ArtiSynth FEM solver, returns the deformed mesh.
"""
import os, socket, csv
import numpy as np

HOST        = os.environ.get("TONGUE_HOST", "127.0.0.1")
PORT        = int(os.environ.get("TONGUE_PORT", "5005"))
FORWARD_DIR = os.environ.get("FORWARD_DIR", r"C:\Users\d11\Project\Tongue_Inverse\forward")


def _recv_line(sock):
    buf = b""
    while not buf.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
    return buf.decode().strip()


def _send(msg):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    s.sendall((msg + "\n").encode())
    line = _recv_line(s)
    s.close()
    return line


def muscle_names():
    """Exciter order expected by muscle_power (query the running server)."""
    return _send("NAMES").split(",")


def load_faces(forward_dir=None):
    """Triangle faces (Nfaces,3) written by the server at startup."""
    p = os.path.join(forward_dir or FORWARD_DIR, "faces.csv")
    F = []
    with open(p) as f:
        for r in csv.reader(f):
            if not r or r[0] == "i0":
                continue
            F.append([int(x) for x in r])
    return np.array(F, dtype=int)


def muscle_power(a, faces=None):
    """activations (list in muscle order, or dict {name:val}) -> (verts (N,3), faces (F,3))."""
    if isinstance(a, dict):
        names = muscle_names()
        a = [float(a.get(n, 0.0)) for n in names]
    line = _send(",".join("%.6f" % float(x) for x in a))
    if line.startswith("ERR") or line == "BYE":
        raise RuntimeError("server: " + line)
    verts = np.array([[float(c) for c in p.split(",")] for p in line.split(";")])
    if faces is None:
        try:
            faces = load_faces()
        except Exception:
            faces = None
    return verts, faces


def save_obj(verts, faces, path):
    with open(path, "w") as f:
        for v in verts:
            f.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        if faces is not None:
            for t in faces:
                f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))


def quit_server():
    return _send("QUIT")


if __name__ == "__main__":
    print("muscles:", muscle_names())
    v, f = muscle_power([0.3] + [0.0] * 10)
    print("verts:", v.shape, "faces:", None if f is None else f.shape)
    save_obj(v, f, "pose_test.obj")
    print("saved pose_test.obj")
