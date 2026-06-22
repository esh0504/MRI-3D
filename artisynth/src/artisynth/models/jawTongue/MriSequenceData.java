package artisynth.models.jawTongue;

import java.util.ArrayList;
import java.util.LinkedHashMap;

import maspack.matrix.Point2d;
import maspack.matrix.Point3d;

/**
 * In-memory representation of a single midsagittal MRI sequence.
 */
public class MriSequenceData {

   public static class PhoneInterval {
      public final double startTime;
      public final double stopTime;
      public final String label;

      public PhoneInterval (double startTime, double stopTime, String label) {
         this.startTime = startTime;
         this.stopTime = stopTime;
         this.label = label;
      }

      public boolean contains (double t) {
         return t >= startTime && t <= stopTime;
      }
   }

   public static class RegistrationPair {
      public final String label;
      public final Point2d imagePoint;
      public final Point2d modelPoint;

      public RegistrationPair (String label, Point2d imagePoint, Point2d modelPoint) {
         this.label = label;
         this.imagePoint = imagePoint;
         this.modelPoint = modelPoint;
      }
   }

   public static class FrameSample {
      public final int frameIndex;
      public final double startTime;
      public final double stopTime;
      public String phoneLabel;
      public final LinkedHashMap<String, Point2d> landmarks2d =
         new LinkedHashMap<String, Point2d>();
      public final LinkedHashMap<String, Point3d> landmarks3d =
         new LinkedHashMap<String, Point3d>();
      public final LinkedHashMap<String, ArrayList<Point2d>> contours2d =
         new LinkedHashMap<String, ArrayList<Point2d>>();
      public final LinkedHashMap<String, ArrayList<Point3d>> contours3d =
         new LinkedHashMap<String, ArrayList<Point3d>>();

      public FrameSample (int frameIndex, double startTime, double stopTime) {
         this.frameIndex = frameIndex;
         this.startTime = startTime;
         this.stopTime = stopTime;
      }

      public double getMidTime() {
         return 0.5 * (startTime + stopTime);
      }

      public ArrayList<Point2d> getContour2d (String structure) {
         return contours2d.get(structure);
      }

      public ArrayList<Point3d> getContour3d (String structure) {
         return contours3d.get(structure);
      }

      public Point2d getLandmark2d (String label) {
         return landmarks2d.get(label);
      }

      public Point3d getLandmark3d (String label) {
         return landmarks3d.get(label);
      }

      public Point3d getContourCentroid3d (String structure) {
         ArrayList<Point3d> pts = contours3d.get(structure);
         if (pts == null || pts.isEmpty()) {
            return null;
         }
         Point3d centroid = new Point3d();
         for (Point3d p : pts) {
            centroid.add(p);
         }
         centroid.scale(1.0 / pts.size());
         return centroid;
      }
   }

   public final String clipId;
   public final double frameRate;
   public final ArrayList<FrameSample> frames = new ArrayList<FrameSample>();
   public final ArrayList<PhoneInterval> phoneIntervals = new ArrayList<PhoneInterval>();
   public final ArrayList<RegistrationPair> registrationPairs =
      new ArrayList<RegistrationPair>();

   public MriSequenceData (String clipId, double frameRate) {
      this.clipId = clipId;
      this.frameRate = frameRate;
   }

   public FrameSample getFrame (int frameIndex) {
      if (frameIndex < 1 || frameIndex > frames.size()) {
         return null;
      }
      return frames.get(frameIndex-1);
   }

   public String getPhoneAtTime (double t) {
      for (PhoneInterval interval : phoneIntervals) {
         if (interval.contains(t)) {
            return interval.label;
         }
      }
      return null;
   }
}
