# -*- coding: utf-8 -*-
# ArtiSynth forward-model SOCKET SERVER (Jython).
# Run via Scripts -> Run Script...  OR  script('.../forward_server.py',[])
# Uses ONLY Java APIs (ArtiSynth Jython has no Python stdlib like os).
# Receive activation vector -> reset -> setExcitation -> settle -> return mesh.
PORT        = 5005
SETTLE_T    = 0.4
FORWARD_DIR = r"C:\Users\d11\Project\Tongue_Inverse\forward"
def find_tongue():
    def rec(m):
        try:
            if hasattr(m, "getMuscleExciters") and m.getMuscleExciters().size() > 0:
                return m
        except:
            pass
        try:
            for c in m.models():
                r = rec(c)
                if r is not None:
                    return r
        except:
            pass
        return None
    for top in root().models():
        r = rec(top)
        if r is not None:
            return r
    return None
def deactivate_probes():
    for ip in root().getInputProbes():
        try:
            ip.setActive(False)
        except:
            ip.setActive(0)
def export_static(tongue, names):
    from java.io import File, FileWriter, PrintWriter
    d = File(FORWARD_DIR)
    if not d.exists():
        d.mkdirs()
    sm = tongue.getSurfaceMesh()
    fo = PrintWriter(FileWriter(File(d, "faces.csv")))
    fo.println("i0,i1,i2")
    for f in sm.getFaces():
        vi = f.getVertexIndices()
        fo.println("%d,%d,%d" % (vi[0], vi[1], vi[2]))
    fo.close()
    fn = PrintWriter(FileWriter(File(d, "muscle_names.txt")))
    for nm in names:
        fn.println(nm)
    fn.close()
    fr = PrintWriter(FileWriter(File(d, "rest_verts.csv")))
    fr.println("x,y,z")
    for v in sm.getVertices():
        p = v.getPosition()
        fr.println("%.6f,%.6f,%.6f" % (p.x, p.y, p.z))
    fr.close()
    print "exported faces/names/rest_verts to %s" % FORWARD_DIR
def serve():
    from java.net import ServerSocket
    from java.io import BufferedReader, InputStreamReader, PrintWriter
    loadModel("artisynth.models.tongue3d.HexTongueDemo")
    tongue = find_tongue()
    if tongue is None:
        print "ERROR: no FemMuscleModel with exciters found"
        return
    mech = root().models().get(0)
    try:
        mech.setGravity(0, 0, 0)
        tongue.setGravity(0, 0, 0)
    except:
        pass
    deactivate_probes()
    exciters = list(tongue.getMuscleExciters())
    names = [e.getName() for e in exciters]
    sm = tongue.getSurfaceMesh()
    export_static(tongue, names)
    print "Forward server: %d exciters, %d verts. order: %s" % (len(exciters), sm.numVertices(), ",".join(names))
    ss = ServerSocket(PORT)
    print "Listening on port %d ... (send QUIT to stop)" % PORT
    running = True
    while running:
        sock = ss.accept()
        try:
            rd = BufferedReader(InputStreamReader(sock.getInputStream()))
            pw = PrintWriter(sock.getOutputStream(), True)
            line = rd.readLine()
            if line is None:
                sock.close(); continue
            line = line.strip()
            if line == "QUIT":
                pw.println("BYE"); sock.close(); running = False; continue
            if line == "NAMES":
                pw.println(",".join(names)); sock.close(); continue
            vals = [float(x) for x in line.split(",") if x != ""]
            reset()
            deactivate_probes()
            for i in range(len(exciters)):
                exciters[i].setExcitation(vals[i] if i < len(vals) else 0.0)
            addBreakPoint(SETTLE_T)
            run()
            waitForStop()
            parts = []
            for v in sm.getVertices():
                p = v.getPosition()
                parts.append("%.6f,%.6f,%.6f" % (p.x, p.y, p.z))
            pw.println(";".join(parts))
            sock.close()
        except Exception, e:
            try:
                pw.println("ERR " + str(e)); sock.close()
            except:
                pass
            print "request error:", e
    ss.close()
    reset()
    print "Forward server stopped."
serve()
