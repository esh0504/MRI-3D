package artisynth.models.jawTongue;

import java.util.ArrayList;

import maspack.matrix.AffineTransform2d;
import maspack.matrix.Point2d;
import maspack.matrix.Point3d;

/**
 * Maps image-space midsagittal coordinates to ArtiSynth model x-z coordinates.
 */
public class MriRegistration2d {

   private final AffineTransform2d myImageToModel = new AffineTransform2d();

   public void fit (ArrayList<MriSequenceData.RegistrationPair> pairs) {
      ArrayList<Point2d> model = new ArrayList<Point2d>();
      ArrayList<Point2d> image = new ArrayList<Point2d>();
      for (MriSequenceData.RegistrationPair pair : pairs) {
         model.add(pair.modelPoint);
         image.add(pair.imagePoint);
      }
      myImageToModel.fit(model, image);
   }

   public Point2d transformToModel2d (Point2d imagePoint) {
      Point2d out = new Point2d();
      out.transform(myImageToModel, imagePoint);
      return out;
   }

   public Point3d transformToModel3d (Point2d imagePoint) {
      Point2d model = transformToModel2d(imagePoint);
      return new Point3d(model.x, 0.0, model.y);
   }

   public Point3d transformToModel3d (double x, double y) {
      return transformToModel3d(new Point2d(x, y));
   }
}
