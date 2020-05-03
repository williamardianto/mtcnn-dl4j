import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.nd4j.linalg.io.ClassPathResource;

import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.circle;
import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;

public class Example {
    public static void main(String[] args) throws Exception {
        String imageFile = new ClassPathResource("image/image.jpg").getFile().getAbsolutePath();
        String outputFile = new ClassPathResource("image").getFile().getAbsolutePath() + "/image-output.jpg";

        Mat image = imread(imageFile);

        MTCNN mtcnn = new MTCNN();

        FaceAnnotation[] faceAnnotations = mtcnn.detectFace(image);

        if(faceAnnotations.length != 0){
            for (FaceAnnotation faceAnnotation : faceAnnotations) {
                FaceAnnotation.BoundingBox bbox = faceAnnotation.getBoundingBox();

                Point x1y1 = new Point(bbox.getX(), bbox.getY());
                Point x2y2 = new Point(bbox.getX()+bbox.getW(), bbox.getY()+bbox.getH());

                rectangle(image, x1y1, x2y2, new Scalar(0, 255, 0, 0));

                for (FaceAnnotation.Landmark lm : faceAnnotation.getLandmarks()) {
                    Point keyPoint = new Point(lm.getPosition().getX(),lm.getPosition().getY());
                    circle(image, keyPoint, 2, new Scalar(0, 255, 0, 0), -1, 0, 0);
                }
            }
        }

        imwrite(outputFile, image);

        imshow("Input Image", image);

//        Press "Esc" to close window
        if (waitKey(0) == 27) {
            destroyAllWindows();
        }
    }
}
