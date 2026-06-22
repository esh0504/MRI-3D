package artisynth.models.tongue3d;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import artisynth.core.femmodels.FemModel;
import artisynth.core.femmodels.FemNode3d;
import artisynth.core.inverse.InverseManager;
import artisynth.core.inverse.TrackingController;
import artisynth.core.inverse.TargetPoint;
import artisynth.core.mechmodels.MuscleExciter;
import artisynth.core.probes.NumericOutputProbe;
import artisynth.core.probes.PositionInputProbe;
import artisynth.core.util.ArtisynthPath;
import artisynth.core.workspace.DriverInterface;
import maspack.interpolation.Interpolation.Order;
import maspack.matrix.Point3d;
import maspack.matrix.Vector3d;
import maspack.properties.PropertyList;
import maspack.render.RenderProps;

import artisynth.models.jawTongue.MriSequenceManifest;
import artisynth.models.jawTongue.MriSequenceIO;
import artisynth.models.jawTongue.MriSequenceData;
import artisynth.models.jawTongue.MriRegistration2d;

/**
 * Tongue-ONLY MRI fitting driver: drives the standalone FEM muscle tongue
 * (HexTongueDemo, metres) from a midsagittal MRI contour, reusing the same
 * manifest / CSV / 2D-registration infrastructure as JawFemMuscleTongueMriDemo
 * but with no jaw or hyoid. Target interpolation is Linear (smoother than the
 * jaw demo's Step) to reduce first-frame blow-up at low frame rates.
 *
 * Run:  -model artisynth.models.tongue3d.FemTongueMriDemo [ /path/mri_fit_tongue.properties ]
 * or put mri_fit_tongue.properties in the working dir and Load from class.
 * Output: computed_excitations (=activations), tracked/source positions.
 */
public class FemTongueMriDemo extends HexTongueDemo {

   public static PropertyList myProps =
      new PropertyList (FemTongueMriDemo.class, HexTongueDemo.class);
   static {
      myProps.addReadOnly (
         "tongueFitError", "mean distance from tongue source nodes to MRI targets");
   }
   public PropertyList getAllPropertyInfo() { return myProps; }

   public static final String DEFAULT_MANIFEST_FILE = "mri_fit_tongue.properties";

   // dorsal tongue surface nodes (same numbering as TongueInvDemo / the jaw demo)
   public static final int[] DEFAULT_TONGUE_TARGET_NODES =
      new int[] { 926, 919, 912, 873, 869, 908, 859, 927, 928, 846, 847 };

   protected File myManifestFile;
   protected MriSequenceManifest myManifest;
   protected MriSequenceData mySequence;
   protected MriRegistration2d myRegistration;
   protected TrackingController myTrackingController;
   protected ArrayList<TargetPoint> myTongueTargets = new ArrayList<TargetPoint>();

   public FemTongueMriDemo () {
      super();
   }

   @Override
   public void build (String[] args) throws IOException {
      super.build (args);
      tongue.setGravity (0, 0, 0);
      mech.setGravity (0, 0, 0);
      // AUTO incompressibility blows up under inverse load (inverted elements)
      tongue.setIncompressible (FemModel.IncompMethod.OFF);
      myManifestFile = resolveManifestFile (args);
   }

   @Override
   public void attach (DriverInterface driver) {
      super.attach (driver);
      removeAllInputProbes();
      removeAllOutputProbes();
      try {
         configureFromManifest();
      }
      catch (Exception e) {
         System.err.println ("Tongue MRI fitting setup failed: " + e.getMessage());
         e.printStackTrace();
      }
   }

   protected void configureFromManifest () throws Exception {
      if (myManifestFile == null || !myManifestFile.exists()) {
         System.out.println (
            "FemTongueMriDemo: manifest not found, expected "
            + DEFAULT_MANIFEST_FILE + " in the working dir or via "
            + "-Dartisynth.mriManifest=/path or model arg [ /path ]");
         return;
      }
      myManifest = MriSequenceManifest.load (myManifestFile);
      if (myManifest.getManifestFile().getParentFile().exists()) {
         ArtisynthPath.setWorkingDir (myManifest.getManifestFile().getParentFile());
      }
      mySequence = MriSequenceIO.loadSequence (myManifest);
      myRegistration = new MriRegistration2d();
      if (mySequence.registrationPairs.size() >= 3) {
         myRegistration.fit (mySequence.registrationPairs);
      }
      MriSequenceIO.applyRegistration (mySequence, myRegistration);

      setupTrackingController();
      setupTargetProbe();
      setupMetricProbe();

      setMaxStepSize (0.001);
      setAdaptiveStepping (false);
   }

   protected void setupTrackingController () {
      myTrackingController = new TrackingController (mech, "mriTracking");
      myTongueTargets.clear();
      for (int idx : getTongueTargetNodeNumbers()) {
         FemNode3d node = tongue.getNode (idx);
         if (node == null) continue;
         TargetPoint t =
            myTrackingController.addPointTarget (node, myManifest.getTongueTargetWeight());
         t.setSubWeights (new Vector3d (1, 0, 1));   // track x,z; ignore lateral y
         myTongueTargets.add (t);
      }
      for (MuscleExciter ex : tongue.getMuscleExciters()) {
         myTrackingController.addExciter (ex);
      }
      myTrackingController.setTargetsPointRadius (0.001);   // 1 mm in metres
      myTrackingController.setNormalizeH (true);
      myTrackingController.setMaxExcitationJump (myManifest.getMaxExcitationJump());
      myTrackingController.addRegularizationTerms (
         myManifest.getL2Regularization(), myManifest.getDampingRegularization());
      myTrackingController.setProbeDuration (getSequenceDuration());
      myTrackingController.createProbesAndPanel (this);
      addController (myTrackingController);

      InverseManager.setProbeFileName (
         this, InverseManager.ProbeID.TARGET_POSITIONS, "tongue_target_positions.txt");
      InverseManager.setProbeFileName (
         this, InverseManager.ProbeID.TRACKED_POSITIONS, "tongue_tracked_positions.txt");
      InverseManager.setProbeFileName (
         this, InverseManager.ProbeID.SOURCE_POSITIONS, "tongue_source_positions.txt");
      InverseManager.setProbeFileName (
         this, InverseManager.ProbeID.COMPUTED_EXCITATIONS, "subject1_computed_excitations.txt");
      RenderProps.setPointRadius (tongue, 0.001);
   }

   protected void setupTargetProbe () {
      if (myTrackingController == null || mySequence == null) return;
      PositionInputProbe probe = InverseManager.findPositionInputProbe (
         this, InverseManager.ProbeID.TARGET_POSITIONS);
      if (probe == null) return;
      probe.setInterpolationOrder (Order.Linear);   // smoother than Step at 5 fps
      probe.setActive (true);
      for (MriSequenceData.FrameSample frame : mySequence.frames) {
         double t = frame.startTime;
         ArrayList<Point3d> contour = frame.getContour3d (myManifest.getTongueStructure());
         if (contour == null || contour.isEmpty() || myTongueTargets.isEmpty()) continue;
         ArrayList<Point3d> samples =
            MriSequenceIO.resampleContour (contour, myTongueTargets.size());
         for (int i = 0; i < myTongueTargets.size() && i < samples.size(); ++i) {
            probe.setPointData (myTongueTargets.get(i), t, samples.get(i));
         }
      }
   }

   protected void setupMetricProbe () {
      double dur = getSequenceDuration();
      NumericOutputProbe p =
         new NumericOutputProbe (this, "tongueFitError", "tongue_fit_error.txt", 0.05);
      p.setName ("tongue fit error");
      p.setStartStopTimes (0, dur);
      addOutputProbe (p);
   }

   protected ArrayList<Integer> getTongueTargetNodeNumbers () {
      ArrayList<Integer> nums = myManifest.getTongueTargetNodeNumbers();
      if (nums == null || nums.isEmpty()) {
         nums = new ArrayList<Integer>();
         for (int n : DEFAULT_TONGUE_TARGET_NODES) nums.add (n);
      }
      return nums;
   }

   protected double getSequenceDuration () {
      if (mySequence == null || mySequence.frames.isEmpty()) return 1.0;
      return mySequence.frames.get (mySequence.frames.size()-1).stopTime;
   }

   public double getTongueFitError () {
      if (myTongueTargets == null || myTongueTargets.isEmpty()) return 0;
      double sum = 0; int n = 0;
      for (TargetPoint t : myTongueTargets) {
         FemNode3d src = (FemNode3d) t.getSourceComp();
         if (src == null) continue;
         sum += src.getPosition().distance (t.getPosition()); ++n;
      }
      return n > 0 ? sum / n : 0;
   }

   protected File resolveManifestFile (String[] args) {
      if (args != null && args.length > 0 && args[0] != null && args[0].trim().length() > 0) {
         File f = new File (args[0].trim());
         if (!f.isAbsolute()) f = new File (ArtisynthPath.getWorkingDir(), args[0].trim());
         return f;
      }
      String sys = System.getProperty ("artisynth.mriManifest");
      if (sys != null && sys.trim().length() > 0) return new File (sys.trim());
      return new File (ArtisynthPath.getWorkingDir(), DEFAULT_MANIFEST_FILE);
   }
}
