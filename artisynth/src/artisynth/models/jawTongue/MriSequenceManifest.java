package artisynth.models.jawTongue;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Properties;

/**
 * Configuration for a single MRI fitting clip. The format is deliberately kept
 * simple so that masks/landmarks exported from external tooling can be mapped
 * into ArtiSynth without adding JSON dependencies.
 */
public class MriSequenceManifest {

   public static final String DEFAULT_FRAME_TIER = "frame_visual_manual";
   public static final String DEFAULT_PHONE_TIER = "phone_visual_final";
   public static final String DEFAULT_TONGUE_STRUCTURE = "tongue";
   public static final String DEFAULT_JAW_STRUCTURE = "jaw";
   public static final String DEFAULT_HYOID_STRUCTURE = "hyoid";
   public static final String DEFAULT_PALATE_STRUCTURE = "palate";
   public static final String DEFAULT_MAXILLA_STRUCTURE = "maxilla";

   File manifestFile;
   String clipId;
   File eafFile;
   String frameTier = DEFAULT_FRAME_TIER;
   String phoneTier = DEFAULT_PHONE_TIER;
   double frameRate = 20.0;
   int frameCount = 0;

   File landmarkCsv;
   File contourCsv;
   File registrationCsv;

   File frameImageDir;
   String frameImagePrefix = "frame";
   String frameImageExt = ".png";

   String tongueStructure = DEFAULT_TONGUE_STRUCTURE;
   String jawStructure = DEFAULT_JAW_STRUCTURE;
   String hyoidStructure = DEFAULT_HYOID_STRUCTURE;
   String palateStructure = DEFAULT_PALATE_STRUCTURE;
   String maxillaStructure = DEFAULT_MAXILLA_STRUCTURE;

   String jawAnchorLabel = "jaw";
   String jawDirectionLabel = "";
   String hyoidAnchorLabel = "hyoid";

   double tongueTargetWeight = 1.0;
   double jawTargetWeight = 0.4;
   double hyoidTargetWeight = 0.2;
   double l2Regularization = 0.01;
   double dampingRegularization = 0.01;
   double maxExcitationJump = 0.1;

   ArrayList<Integer> tongueTargetNodeNumbers = new ArrayList<Integer>();

   public static MriSequenceManifest load (File file) throws IOException {
      Properties props = new Properties();
      FileInputStream fis = new FileInputStream(file);
      try {
         props.load(fis);
      }
      finally {
         fis.close();
      }

      MriSequenceManifest manifest = new MriSequenceManifest();
      manifest.manifestFile = file.getCanonicalFile();
      manifest.clipId = props.getProperty("clipId", stripExt(file.getName()));
      manifest.eafFile = manifest.resolveOptionalFile(props, "eafFile");
      manifest.frameTier = props.getProperty("frameTier", DEFAULT_FRAME_TIER);
      manifest.phoneTier = props.getProperty("phoneTier", DEFAULT_PHONE_TIER);
      manifest.frameRate =
         parseDouble(props.getProperty("frameRate"), manifest.frameRate);
      manifest.frameCount =
         parseInt(props.getProperty("frameCount"), manifest.frameCount);

      manifest.landmarkCsv = manifest.resolveOptionalFile(props, "landmarkCsv");
      manifest.contourCsv = manifest.resolveOptionalFile(props, "contourCsv");
      manifest.registrationCsv =
         manifest.resolveOptionalFile(props, "registrationCsv");

      manifest.frameImageDir =
         manifest.resolveOptionalFile(props, "frameImageDir");
      manifest.frameImagePrefix =
         props.getProperty("frameImagePrefix", manifest.frameImagePrefix);
      manifest.frameImageExt =
         props.getProperty("frameImageExt", manifest.frameImageExt);

      manifest.tongueStructure =
         props.getProperty("tongueStructure", manifest.tongueStructure);
      manifest.jawStructure =
         props.getProperty("jawStructure", manifest.jawStructure);
      manifest.hyoidStructure =
         props.getProperty("hyoidStructure", manifest.hyoidStructure);
      manifest.palateStructure =
         props.getProperty("palateStructure", manifest.palateStructure);
      manifest.maxillaStructure =
         props.getProperty("maxillaStructure", manifest.maxillaStructure);

      manifest.jawAnchorLabel =
         props.getProperty("jawAnchorLabel", manifest.jawAnchorLabel);
      manifest.jawDirectionLabel =
         props.getProperty("jawDirectionLabel", manifest.jawDirectionLabel);
      manifest.hyoidAnchorLabel =
         props.getProperty("hyoidAnchorLabel", manifest.hyoidAnchorLabel);

      manifest.tongueTargetWeight =
         parseDouble(
            props.getProperty("tongueTargetWeight"),
            manifest.tongueTargetWeight);
      manifest.jawTargetWeight =
         parseDouble(
            props.getProperty("jawTargetWeight"),
            manifest.jawTargetWeight);
      manifest.hyoidTargetWeight =
         parseDouble(
            props.getProperty("hyoidTargetWeight"),
            manifest.hyoidTargetWeight);
      manifest.l2Regularization =
         parseDouble(
            props.getProperty("l2Regularization"),
            manifest.l2Regularization);
      manifest.dampingRegularization =
         parseDouble(
            props.getProperty("dampingRegularization"),
            manifest.dampingRegularization);
      manifest.maxExcitationJump =
         parseDouble(
            props.getProperty("maxExcitationJump"),
            manifest.maxExcitationJump);

      manifest.tongueTargetNodeNumbers =
         parseIntList(props.getProperty("tongueTargetNodes", ""));

      return manifest;
   }

   private File resolveRequiredFile (Properties props, String key)
      throws IOException {
      File file = resolveOptionalFile(props, key);
      if (file == null) {
         throw new IOException("Missing required manifest property: " + key);
      }
      return file;
   }

   private File resolveOptionalFile (Properties props, String key)
      throws IOException {
      String value = props.getProperty(key, "").trim();
      if (value.length() == 0) {
         return null;
      }
      File file = new File(value);
      if (!file.isAbsolute()) {
         file = new File(manifestFile.getParentFile(), value);
      }
      return file.getCanonicalFile();
   }

   public File getManifestFile() {
      return manifestFile;
   }

   public String getClipId() {
      return clipId;
   }

   public File getEafFile() {
      return eafFile;
   }

   public String getFrameTier() {
      return frameTier;
   }

   public String getPhoneTier() {
      return phoneTier;
   }

   public double getFrameRate() {
      return frameRate;
   }

   public int getFrameCount() {
      return frameCount;
   }

   public File getLandmarkCsv() {
      return landmarkCsv;
   }

   public File getContourCsv() {
      return contourCsv;
   }

   public File getRegistrationCsv() {
      return registrationCsv;
   }

   public File getFrameImageDir() {
      return frameImageDir;
   }

   public String getFrameImagePrefix() {
      return frameImagePrefix;
   }

   public String getFrameImageExt() {
      return frameImageExt;
   }

   public String getTongueStructure() {
      return tongueStructure;
   }

   public String getJawStructure() {
      return jawStructure;
   }

   public String getHyoidStructure() {
      return hyoidStructure;
   }

   public String getPalateStructure() {
      return palateStructure;
   }

   public String getMaxillaStructure() {
      return maxillaStructure;
   }

   public String getJawAnchorLabel() {
      return jawAnchorLabel;
   }

   public String getJawDirectionLabel() {
      return jawDirectionLabel;
   }

   public String getHyoidAnchorLabel() {
      return hyoidAnchorLabel;
   }

   public double getTongueTargetWeight() {
      return tongueTargetWeight;
   }

   public double getJawTargetWeight() {
      return jawTargetWeight;
   }

   public double getHyoidTargetWeight() {
      return hyoidTargetWeight;
   }

   public double getL2Regularization() {
      return l2Regularization;
   }

   public double getDampingRegularization() {
      return dampingRegularization;
   }

   public double getMaxExcitationJump() {
      return maxExcitationJump;
   }

   public ArrayList<Integer> getTongueTargetNodeNumbers() {
      return tongueTargetNodeNumbers;
   }

   private static double parseDouble (String value, double defaultValue) {
      if (value == null || value.trim().length() == 0) {
         return defaultValue;
      }
      return Double.parseDouble(value.trim());
   }

   private static int parseInt (String value, int defaultValue) {
      if (value == null || value.trim().length() == 0) {
         return defaultValue;
      }
      return Integer.parseInt(value.trim());
   }

   private static ArrayList<Integer> parseIntList (String value) {
      ArrayList<Integer> values = new ArrayList<Integer>();
      if (value == null || value.trim().length() == 0) {
         return values;
      }
      String[] tokens = value.split(",");
      for (String token : tokens) {
         String trimmed = token.trim();
         if (trimmed.length() > 0) {
            values.add(Integer.parseInt(trimmed));
         }
      }
      return values;
   }

   private static String stripExt (String name) {
      int idx = name.lastIndexOf('.');
      if (idx == -1) {
         return name;
      }
      return name.substring(0, idx);
   }
}
