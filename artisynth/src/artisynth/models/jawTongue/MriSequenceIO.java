package artisynth.models.jawTongue;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import maspack.matrix.Point2d;
import maspack.matrix.Point3d;

/**
 * Reads clip-level MRI fitting inputs exported as ELAN plus simple CSV files.
 */
public class MriSequenceIO {

   public static MriSequenceData loadSequence (MriSequenceManifest manifest)
      throws Exception {

      MriSequenceData data;
      if (manifest.getEafFile() != null && manifest.getEafFile().exists()) {
         data =
            loadEaf(manifest.getEafFile(), manifest.getClipId(),
               manifest.getFrameRate(), manifest.getFrameTier(),
               manifest.getPhoneTier());
      }
      else {
         data =
            createVideoTimedSequence(
               manifest.getClipId(), manifest.getFrameRate(),
               manifest.getFrameCount());
      }

      if (manifest.getLandmarkCsv() != null && manifest.getLandmarkCsv().exists()) {
         loadLandmarks(manifest.getLandmarkCsv(), data);
      }
      if (manifest.getContourCsv() != null && manifest.getContourCsv().exists()) {
         loadContours(manifest.getContourCsv(), data);
      }
      if (manifest.getRegistrationCsv() != null &&
          manifest.getRegistrationCsv().exists()) {
         loadRegistrationPairs(manifest.getRegistrationCsv(), data);
      }
      return data;
   }

   private static MriSequenceData createVideoTimedSequence (
      String clipId, double frameRate, int frameCount) {

      if (frameRate <= 0) {
         throw new IllegalArgumentException(
            "frameRate must be positive when eafFile is not provided");
      }
      if (frameCount <= 0) {
         throw new IllegalArgumentException(
            "frameCount must be positive when eafFile is not provided");
      }

      MriSequenceData data = new MriSequenceData(clipId, frameRate);
      double dt = 1.0 / frameRate;
      for (int i=0; i<frameCount; ++i) {
         double start = i * dt;
         double stop = (i+1) * dt;
         data.frames.add(new MriSequenceData.FrameSample(i+1, start, stop));
      }
      return data;
   }

   public static void applyRegistration (
      MriSequenceData data, MriRegistration2d registration) {

      for (MriSequenceData.FrameSample frame : data.frames) {
         frame.landmarks3d.clear();
         frame.contours3d.clear();

         for (Map.Entry<String, Point2d> entry : frame.landmarks2d.entrySet()) {
            frame.landmarks3d.put(
               entry.getKey(), registration.transformToModel3d(entry.getValue()));
         }

         for (Map.Entry<String, ArrayList<Point2d>> entry : frame.contours2d.entrySet()) {
            ArrayList<Point3d> points3d = new ArrayList<Point3d>();
            for (Point2d p : entry.getValue()) {
               points3d.add(registration.transformToModel3d(p));
            }
            frame.contours3d.put(entry.getKey(), points3d);
         }
      }
   }

   private static MriSequenceData loadEaf (
      File eafFile, String clipId, double defaultFrameRate,
      String frameTierId, String phoneTierId) throws Exception {

      DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
      dbf.setNamespaceAware(false);
      DocumentBuilder db = dbf.newDocumentBuilder();
      Document doc = db.parse(eafFile);
      Element root = doc.getDocumentElement();

      HashMap<String,Integer> times = new HashMap<String,Integer>();
      NodeList timeSlots = root.getElementsByTagName("TIME_SLOT");
      for (int i=0; i<timeSlots.getLength(); ++i) {
         Element slot = (Element)timeSlots.item(i);
         String id = slot.getAttribute("TIME_SLOT_ID");
         String value = slot.getAttribute("TIME_VALUE");
         times.put(id, Integer.parseInt(value));
      }

      MriSequenceData data = new MriSequenceData(clipId, defaultFrameRate);
      loadTierIntervals(root, times, frameTierId, data.frames);
      loadPhoneIntervals(root, times, phoneTierId, data.phoneIntervals);

      for (MriSequenceData.FrameSample frame : data.frames) {
         frame.phoneLabel = data.getPhoneAtTime(frame.getMidTime());
      }
      return data;
   }

   private static void loadTierIntervals (
      Element root, HashMap<String,Integer> times, String tierId,
      ArrayList<MriSequenceData.FrameSample> frames) {

      Element tier = findTier(root, tierId);
      if (tier == null) {
         return;
      }
      NodeList annotations = tier.getElementsByTagName("ALIGNABLE_ANNOTATION");
      for (int i=0; i<annotations.getLength(); ++i) {
         Element ann = (Element)annotations.item(i);
         double start =
            times.get(ann.getAttribute("TIME_SLOT_REF1")) / 1000.0;
         double stop =
            times.get(ann.getAttribute("TIME_SLOT_REF2")) / 1000.0;
         frames.add(new MriSequenceData.FrameSample(i+1, start, stop));
      }
   }

   private static void loadPhoneIntervals (
      Element root, HashMap<String,Integer> times, String tierId,
      ArrayList<MriSequenceData.PhoneInterval> intervals) {

      Element tier = findTier(root, tierId);
      if (tier == null) {
         return;
      }
      NodeList annotations = tier.getElementsByTagName("ALIGNABLE_ANNOTATION");
      for (int i=0; i<annotations.getLength(); ++i) {
         Element ann = (Element)annotations.item(i);
         double start =
            times.get(ann.getAttribute("TIME_SLOT_REF1")) / 1000.0;
         double stop =
            times.get(ann.getAttribute("TIME_SLOT_REF2")) / 1000.0;
         String label = getAnnotationValue(ann);
         intervals.add(new MriSequenceData.PhoneInterval(start, stop, label));
      }
   }

   private static Element findTier (Element root, String tierId) {
      NodeList tiers = root.getElementsByTagName("TIER");
      for (int i=0; i<tiers.getLength(); ++i) {
         Element tier = (Element)tiers.item(i);
         if (tierId.equals(tier.getAttribute("TIER_ID"))) {
            return tier;
         }
      }
      return null;
   }

   private static String getAnnotationValue (Element ann) {
      NodeList values = ann.getElementsByTagName("ANNOTATION_VALUE");
      if (values.getLength() == 0) {
         return "";
      }
      Node node = values.item(0);
      return node.getTextContent() == null ? "" : node.getTextContent().trim();
   }

   private static void loadLandmarks (File csv, MriSequenceData data)
      throws IOException {

      CsvHeader header = new CsvHeader(csv);
      String line;
      while ((line = header.reader.readLine()) != null) {
         line = normalizeLine(line);
         if (line == null) {
            continue;
         }
         String[] tokens = splitCsv(line);
         int frame = header.getInt(tokens, "frame");
         String label = header.getString(tokens, "label");
         double x = header.getDouble(tokens, "x");
         double y = header.getDouble(tokens, "y");
         MriSequenceData.FrameSample sample = data.getFrame(frame);
         if (sample != null) {
            sample.landmarks2d.put(label, new Point2d(x, y));
         }
      }
      header.close();
   }

   private static void loadContours (File csv, MriSequenceData data)
      throws IOException {

      CsvHeader header = new CsvHeader(csv);
      String line;
      while ((line = header.reader.readLine()) != null) {
         line = normalizeLine(line);
         if (line == null) {
            continue;
         }
         String[] tokens = splitCsv(line);
         int frame = header.getInt(tokens, "frame");
         String structure = header.getString(tokens, "structure");
         double x = header.getDouble(tokens, "x");
         double y = header.getDouble(tokens, "y");

         MriSequenceData.FrameSample sample = data.getFrame(frame);
         if (sample == null) {
            continue;
         }
         ArrayList<Point2d> contour = sample.contours2d.get(structure);
         if (contour == null) {
            contour = new ArrayList<Point2d>();
            sample.contours2d.put(structure, contour);
         }
         contour.add(new Point2d(x, y));
      }
      header.close();
   }

   private static void loadRegistrationPairs (File csv, MriSequenceData data)
      throws IOException {

      CsvHeader header = new CsvHeader(csv);
      String line;
      while ((line = header.reader.readLine()) != null) {
         line = normalizeLine(line);
         if (line == null) {
            continue;
         }
         String[] tokens = splitCsv(line);
         String label = header.getString(tokens, "label");
         double imageX = header.getDouble(tokens, "imageX");
         double imageY = header.getDouble(tokens, "imageY");
         double modelX = header.getDouble(tokens, "modelX");
         double modelZ = header.getDouble(tokens, "modelZ");
         data.registrationPairs.add(
            new MriSequenceData.RegistrationPair(
               label, new Point2d(imageX, imageY), new Point2d(modelX, modelZ)));
      }
      header.close();
   }

   public static ArrayList<Point3d> resampleContour (
      ArrayList<Point3d> contour, int sampleCount) {

      ArrayList<Point3d> out = new ArrayList<Point3d>();
      if (contour == null || contour.isEmpty() || sampleCount <= 0) {
         return out;
      }
      if (contour.size() == 1 || sampleCount == 1) {
         out.add(new Point3d(contour.get(0)));
         return out;
      }

      double[] cumulative = new double[contour.size()];
      cumulative[0] = 0;
      for (int i=1; i<contour.size(); ++i) {
         cumulative[i] =
            cumulative[i-1] + contour.get(i).distance(contour.get(i-1));
      }
      double total = cumulative[cumulative.length-1];
      if (total == 0) {
         for (int i=0; i<sampleCount; ++i) {
            out.add(new Point3d(contour.get(0)));
         }
         return out;
      }

      for (int i=0; i<sampleCount; ++i) {
         double s = total * i / (sampleCount-1.0);
         int seg = 1;
         while (seg < cumulative.length && cumulative[seg] < s) {
            ++seg;
         }
         if (seg >= cumulative.length) {
            out.add(new Point3d(contour.get(contour.size()-1)));
         }
         else {
            double segLen = cumulative[seg] - cumulative[seg-1];
            double alpha = segLen > 0 ? (s-cumulative[seg-1]) / segLen : 0.0;
            Point3d p0 = contour.get(seg-1);
            Point3d p1 = contour.get(seg);
            Point3d pi = new Point3d();
            pi.combine(1.0-alpha, p0, alpha, p1);
            out.add(pi);
         }
      }
      return out;
   }

   private static String normalizeLine (String line) {
      if (line == null) {
         return null;
      }
      String trimmed = line.trim();
      if (trimmed.length() == 0 || trimmed.startsWith("#")) {
         return null;
      }
      return trimmed;
   }

   private static String[] splitCsv (String line) {
      return line.split("\\s*,\\s*");
   }

   private static class CsvHeader {
      BufferedReader reader;
      HashMap<String,Integer> idxMap = new HashMap<String,Integer>();

      CsvHeader (File file) throws IOException {
         reader = new BufferedReader(new FileReader(file));
         String headerLine = reader.readLine();
         if (headerLine == null) {
            throw new IOException("CSV file is empty: " + file);
         }
         String[] headers = splitCsv(headerLine.trim());
         for (int i=0; i<headers.length; ++i) {
            idxMap.put(headers[i], i);
         }
      }

      int getInt (String[] tokens, String key) {
         return Integer.parseInt(tokens[getIndex(key)]);
      }

      double getDouble (String[] tokens, String key) {
         return Double.parseDouble(tokens[getIndex(key)]);
      }

      String getString (String[] tokens, String key) {
         return tokens[getIndex(key)];
      }

      int getIndex (String key) {
         Integer idx = idxMap.get(key);
         if (idx == null) {
            throw new IllegalArgumentException("Missing CSV column: " + key);
         }
         return idx;
      }

      void close() throws IOException {
         reader.close();
      }
   }
}
