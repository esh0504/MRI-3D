package artisynth.models.jawTongue;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import artisynth.core.driver.Main;
import artisynth.core.femmodels.FemNode3d;
import artisynth.core.inverse.InverseManager;
import artisynth.core.inverse.TargetFrame;
import artisynth.core.inverse.TargetPoint;
import artisynth.core.inverse.TrackingController;
import artisynth.core.mechmodels.MuscleExciter;
import artisynth.core.mechmodels.RigidBody;
import artisynth.core.probes.ImagePlaneProbe;
import artisynth.core.probes.NumericOutputProbe;
import artisynth.core.probes.PositionInputProbe;
import artisynth.core.util.ArtisynthPath;
import artisynth.core.workspace.DriverInterface;
import maspack.interpolation.Interpolation.Order;
import maspack.matrix.Point3d;
import maspack.matrix.RigidTransform3d;
import maspack.matrix.Vector3d;
import maspack.matrix.VectorNd;
import maspack.properties.PropertyList;
import maspack.render.RenderProps;

/**
 * MRI fitting prototype for a single clip using a midsagittal contour/landmark
 * observation model. A clip is configured with a {@code .properties} manifest
 * plus simple CSV exports for tongue/jaw/hyoid/palate structures.
 */
public class JawFemMuscleTongueMriDemo extends JawHyoidFemMuscleTongue {

   public static PropertyList myProps =
      new PropertyList (JawFemMuscleTongueMriDemo.class, JawHyoidFemMuscleTongue.class);

   static {
      myProps.addReadOnly (
         "tongueFitError", "mean distance from tongue source nodes to MRI targets");
      myProps.addReadOnly (
         "jawFitError", "distance from jaw frame to MRI target frame");
      myProps.addReadOnly (
         "hyoidFitError", "distance from hyoid frame to MRI target frame");
      myProps.addReadOnly (
         "meanFitError", "mean of tongue, jaw, and hyoid fit errors");
   }

   @Override
   public PropertyList getAllPropertyInfo() {
      return myProps;
   }

   public static final String DEFAULT_MANIFEST_FILE = "mri_fit.properties";

   public static final int[] DEFAULT_TONGUE_TARGET_NODES =
      new int[] { 926, 919, 912, 873, 869, 908, 859, 927, 928, 846, 847 };

   protected File myManifestFile;
   protected MriSequenceManifest myManifest;
   protected MriSequenceData mySequence;
   protected MriRegistration2d myRegistration;

   protected TrackingController myTrackingController;
   protected ArrayList<TargetPoint> myTongueTargets =
      new ArrayList<TargetPoint>();
   protected TargetFrame myJawTarget;
   protected TargetFrame myHyoidTarget;

   protected RigidTransform3d myJawRestPose;
   protected RigidTransform3d myHyoidRestPose;
   protected Point3d myJawReferencePoint;
   protected Point3d myJawReferenceDirection;
   protected Point3d myHyoidReferencePoint;

   protected double myTongueFitError = 0;
   protected double myJawFitError = 0;
   protected double myHyoidFitError = 0;

   @Override
   public void build (String[] args) throws IOException {
      super.build(args);
      myManifestFile = resolveManifestFile(args);
   }

   @Override
   public void attach (DriverInterface driver) {
      super.attach(driver);

      removeAllInputProbes();
      removeAllOutputProbes();

      try {
         maybeConfigureFromManifest();
      }
      catch (Exception e) {
         System.err.println("MRI fitting setup failed: " + e.getMessage());
         e.printStackTrace();
      }
   }

   protected void maybeConfigureFromManifest() throws Exception {
      if (myManifestFile == null || !myManifestFile.exists()) {
         System.out.println(
            "JawFemMuscleTongueMriDemo: manifest not found, expected "
            + DEFAULT_MANIFEST_FILE + " in the working directory or via "
            + "-Dartisynth.mriManifest=/path/to/file.properties");
         return;
      }

      myManifest = MriSequenceManifest.load(myManifestFile);
      if (myManifest.getManifestFile().getParentFile().exists()) {
         ArtisynthPath.setWorkingDir(myManifest.getManifestFile().getParentFile());
      }

      mySequence = MriSequenceIO.loadSequence(myManifest);
      myRegistration = new MriRegistration2d();
      if (mySequence.registrationPairs.size() >= 3) {
         myRegistration.fit(mySequence.registrationPairs);
      }
      MriSequenceIO.applyRegistration(mySequence, myRegistration);

      setupTrackingController();
      setupTargetProbe();
      setupMriOverlay();
      setupMetricProbes();

      // Use a conservative fixed step for MRI inverse playback to reduce
      // first-frame instability when registration or targets are still rough.
      myJawModel.setMaxStepSize(0.001);
      setMaxStepSize(0.001);
      setAdaptiveStepping(false);
   }

   protected void setupTrackingController() {
      myTrackingController = new TrackingController(myJawModel, "mriTracking");

      myTongueTargets.clear();
      for (int nodeNum : getTongueTargetNodeNumbers()) {
         FemNode3d node = (FemNode3d)tongue.getByNumber(nodeNum);
         if (node == null) {
            continue;
         }
         TargetPoint target =
            myTrackingController.addPointTarget(node, myManifest.getTongueTargetWeight());
         target.setSubWeights(new Vector3d(1, 0, 1));
         myTongueTargets.add(target);
      }

      RigidBody jaw = myJawModel.rigidBodies().get("jaw");
      myJawTarget = null;
      if (jaw != null) {
         myJawTarget =
            myTrackingController.addFrameTarget(jaw, myManifest.getJawTargetWeight());
         myJawTarget.setSubWeights(new VectorNd(new double[] {1, 0, 1, 0, 0, 0}));
         myJawTarget.setAxisLength(10.0);
         myJawRestPose = new RigidTransform3d(jaw.getPose());
      }

      RigidBody hyoid = myJawModel.rigidBodies().get("hyoid");
      myHyoidTarget = null;
      if (hyoid != null) {
         myHyoidTarget =
            myTrackingController.addFrameTarget(hyoid, myManifest.getHyoidTargetWeight());
         myHyoidTarget.setSubWeights(new VectorNd(new double[] {1, 0, 1, 0, 0, 0}));
         myHyoidTarget.setAxisLength(8.0);
         myHyoidRestPose = new RigidTransform3d(hyoid.getPose());
      }

      for (MuscleExciter ex : tongue.getMuscleExciters()) {
         myTrackingController.addExciter(ex);
      }
      for (MuscleExciter ex : myJawModel.getMuscleExciters()) {
         myTrackingController.addExciter(ex);
      }

      myTrackingController.setTargetsPointRadius(1.0);
      myTrackingController.setNormalizeH(true);
      myTrackingController.setMaxExcitationJump(myManifest.getMaxExcitationJump());
      myTrackingController.addRegularizationTerms(
         myManifest.getL2Regularization(), myManifest.getDampingRegularization());
      myTrackingController.setProbeDuration(getSequenceDuration());
      myTrackingController.createProbesAndPanel(this);
      addController(myTrackingController);

      configureProbeFileNames();
      RenderProps.setPointRadius(tongue, 0.8);
   }

   protected void setupTargetProbe() {
      if (myTrackingController == null || mySequence == null) {
         return;
      }

      PositionInputProbe probe =
         InverseManager.findPositionInputProbe(
            this, InverseManager.ProbeID.TARGET_POSITIONS);
      if (probe == null) {
         return;
      }
      probe.setInterpolationOrder(Order.Step);
      probe.setActive(true);

      MriSequenceData.FrameSample reference = mySequence.frames.isEmpty() ?
         null : mySequence.frames.get(0);
      if (reference != null) {
         myJawReferencePoint = getJawObservationPoint(reference);
         myJawReferenceDirection = getJawDirectionPoint(reference);
         myHyoidReferencePoint = getHyoidObservationPoint(reference);
      }

      for (MriSequenceData.FrameSample frame : mySequence.frames) {
         double t = frame.startTime;
         applyTongueTargets(probe, frame, t);
         applyJawTarget(probe, frame, t);
         applyHyoidTarget(probe, frame, t);
      }
   }

   protected void applyTongueTargets (
      PositionInputProbe probe, MriSequenceData.FrameSample frame, double t) {

      ArrayList<Point3d> contour = frame.getContour3d(myManifest.getTongueStructure());
      if (contour == null || contour.isEmpty() || myTongueTargets.isEmpty()) {
         return;
      }
      ArrayList<Point3d> samples =
         MriSequenceIO.resampleContour(contour, myTongueTargets.size());
      for (int i=0; i<myTongueTargets.size() && i<samples.size(); ++i) {
         probe.setPointData(myTongueTargets.get(i), t, samples.get(i));
      }
   }

   protected void applyJawTarget (
      PositionInputProbe probe, MriSequenceData.FrameSample frame, double t) {

      if (myJawTarget == null || myJawRestPose == null || myJawReferencePoint == null) {
         return;
      }
      Point3d jawPoint = getJawObservationPoint(frame);
      if (jawPoint == null) {
         return;
      }
      RigidTransform3d pose = new RigidTransform3d(myJawRestPose);
      pose.p.x += jawPoint.x - myJawReferencePoint.x;
      pose.p.z += jawPoint.z - myJawReferencePoint.z;

      Point3d jawDir = getJawDirectionPoint(frame);
      if (jawDir != null && myJawReferenceDirection != null) {
         double a0 = Math.atan2(
            myJawReferenceDirection.z - myJawReferencePoint.z,
            myJawReferenceDirection.x - myJawReferencePoint.x);
         double a1 = Math.atan2(
            jawDir.z - jawPoint.z,
            jawDir.x - jawPoint.x);
         pose.R.mulRotY(a1-a0);
      }
      probe.setFrameData(myJawTarget, t, pose);
   }

   protected void applyHyoidTarget (
      PositionInputProbe probe, MriSequenceData.FrameSample frame, double t) {

      if (myHyoidTarget == null || myHyoidRestPose == null ||
          myHyoidReferencePoint == null) {
         return;
      }
      Point3d hyoidPoint = getHyoidObservationPoint(frame);
      if (hyoidPoint == null) {
         return;
      }
      RigidTransform3d pose = new RigidTransform3d(myHyoidRestPose);
      pose.p.x += hyoidPoint.x - myHyoidReferencePoint.x;
      pose.p.z += hyoidPoint.z - myHyoidReferencePoint.z;
      probe.setFrameData(myHyoidTarget, t, pose);
   }

   protected void setupMriOverlay() {
      if (myManifest == null || myManifest.getFrameImageDir() == null) {
         return;
      }
      File dir = myManifest.getFrameImageDir();
      if (!dir.exists()) {
         return;
      }
      ImagePlaneProbe probe =
         new ImagePlaneProbe(
            myJawModel,
            dir.getAbsolutePath(),
            myManifest.getFrameImagePrefix() + myManifest.getFrameImageExt(),
            myManifest.getFrameRate(),
            0,
            getSequenceDuration(),
            1.0);
      probe.setName("MRI Frames");
      addInputProbe(probe);
   }

   protected void setupMetricProbes() {
      double duration = getSequenceDuration();
      addMetricProbe("tongue fit error", "tongueFitError",
         clipPrefix("tongue_fit_error.txt"), duration);
      addMetricProbe("jaw fit error", "jawFitError",
         clipPrefix("jaw_fit_error.txt"), duration);
      addMetricProbe("hyoid fit error", "hyoidFitError",
         clipPrefix("hyoid_fit_error.txt"), duration);
      addMetricProbe("mean fit error", "meanFitError",
         clipPrefix("mean_fit_error.txt"), duration);
   }

   protected void addMetricProbe (
      String name, String propName, String fileName, double duration) {
      NumericOutputProbe probe = new NumericOutputProbe(this, propName, fileName, 0.05);
      probe.setName(name);
      probe.setStartStopTimes(0, duration);
      addOutputProbe(probe);
   }

   protected void configureProbeFileNames() {
      InverseManager.setProbeFileName(
         this, InverseManager.ProbeID.TARGET_POSITIONS,
         clipPrefix("target_positions.txt"));
      InverseManager.setProbeFileName(
         this, InverseManager.ProbeID.TRACKED_POSITIONS,
         clipPrefix("tracked_positions.txt"));
      InverseManager.setProbeFileName(
         this, InverseManager.ProbeID.SOURCE_POSITIONS,
         clipPrefix("source_positions.txt"));
      InverseManager.setProbeFileName(
         this, InverseManager.ProbeID.COMPUTED_EXCITATIONS,
         clipPrefix("computed_excitations.txt"));
      InverseManager.setProbeFileName(
         this, InverseManager.ProbeID.INPUT_EXCITATIONS,
         clipPrefix("input_excitations.txt"));
   }

   protected ArrayList<Integer> getTongueTargetNodeNumbers() {
      ArrayList<Integer> numbers = myManifest.getTongueTargetNodeNumbers();
      if (numbers == null || numbers.isEmpty()) {
         numbers = new ArrayList<Integer>();
         for (int nodeNum : DEFAULT_TONGUE_TARGET_NODES) {
            numbers.add(nodeNum);
         }
      }
      return numbers;
   }

   protected Point3d getJawObservationPoint (MriSequenceData.FrameSample frame) {
      Point3d point = frame.getLandmark3d(myManifest.getJawAnchorLabel());
      if (point != null) {
         return point;
      }
      return frame.getContourCentroid3d(myManifest.getJawStructure());
   }

   protected Point3d getJawDirectionPoint (MriSequenceData.FrameSample frame) {
      String label = myManifest.getJawDirectionLabel();
      if (label == null || label.length() == 0) {
         return null;
      }
      return frame.getLandmark3d(label);
   }

   protected Point3d getHyoidObservationPoint (MriSequenceData.FrameSample frame) {
      Point3d point = frame.getLandmark3d(myManifest.getHyoidAnchorLabel());
      if (point != null) {
         return point;
      }
      return frame.getContourCentroid3d(myManifest.getHyoidStructure());
   }

   protected double getSequenceDuration() {
      if (mySequence == null || mySequence.frames.isEmpty()) {
         return 1.0;
      }
      return mySequence.frames.get(mySequence.frames.size()-1).stopTime;
   }

   protected String clipPrefix (String suffix) {
      if (myManifest == null) {
         return suffix;
      }
      return myManifest.getClipId() + "_" + suffix;
   }

   protected File resolveManifestFile (String[] args) {
      if (args != null && args.length > 0 && args[0] != null &&
          args[0].trim().length() > 0) {
         File file = new File(args[0].trim());
         if (!file.isAbsolute()) {
            file = new File(ArtisynthPath.getWorkingDir(), args[0].trim());
         }
         return file;
      }

      String sysProp = System.getProperty("artisynth.mriManifest");
      if (sysProp != null && sysProp.trim().length() > 0) {
         return new File(sysProp.trim());
      }

      return new File(ArtisynthPath.getWorkingDir(), DEFAULT_MANIFEST_FILE);
   }

   public double getTongueFitError() {
      myTongueFitError = computeTongueFitError();
      return myTongueFitError;
   }

   public double getJawFitError() {
      myJawFitError = computeFrameFitError(myJawTarget);
      return myJawFitError;
   }

   public double getHyoidFitError() {
      myHyoidFitError = computeFrameFitError(myHyoidTarget);
      return myHyoidFitError;
   }

   public double getMeanFitError() {
      return (getTongueFitError() + getJawFitError() + getHyoidFitError()) / 3.0;
   }

   protected double computeTongueFitError() {
      if (myTongueTargets == null || myTongueTargets.isEmpty()) {
         return 0;
      }
      double sum = 0;
      int count = 0;
      for (TargetPoint target : myTongueTargets) {
         FemNode3d src = (FemNode3d)target.getSourceComp();
         if (src == null) {
            continue;
         }
         sum += src.getPosition().distance(target.getPosition());
         ++count;
      }
      return count > 0 ? sum / count : 0;
   }

   protected double computeFrameFitError (TargetFrame target) {
      if (target == null || target.getSourceComp() == null) {
         return 0;
      }
      return target.getSourceComp().getPosition().distance(target.getPosition());
   }

   public String getPhoneLabel() {
      if (mySequence == null) {
         return "";
      }
      double t = Main.getMain().getTime();
      String label = mySequence.getPhoneAtTime(t);
      return label == null ? "" : label;
   }
}
