import org.apache.commons.io.IOUtils;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_core.Point;

import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.io.Assert;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.tensorflow.framework.ConfigProto;

import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.*;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.*;
import java.util.List;


public class MTCNN {
    private static final int minFaceSize = 20;
    private static final double scaleFactor = 0.709;
    private static final List<Double> stepsTreshold = Arrays.asList(0.6, 0.7, 0.7);

    public enum nmsMethod {Min, Union}

    private static final NativeImageLoader loader = new NativeImageLoader();

    private static GraphRunner pNet;
    private static GraphRunner rNet;
    private static GraphRunner oNet;

    public MTCNN() throws Exception {
        Nd4j.setDefaultDataTypes(DataType.FLOAT, DataType.FLOAT);

        pNet = createGraphRunner("model/pnet.pb", "input_1:0","conv2d_5/BiasAdd:0","softmax_1/truediv:0");
        rNet = createGraphRunner("model/rnet.pb", "input_2:0","dense_3/BiasAdd:0","softmax_2/Softmax:0");
        oNet = createGraphRunner("model/onet.pb", "input_3:0","dense_6/BiasAdd:0","dense_7/BiasAdd:0","softmax_3/Softmax:0");
    }

    private GraphRunner createGraphRunner(String tensorflowModelUri, String inputName, String... outputName) {
        try{
            ConfigProto configProto = ConfigProto.newBuilder()
                    .setInterOpParallelismThreads(8)
                    .build();

            return GraphRunner.builder()
                    .graphBytes(IOUtils.toByteArray(new ClassPathResource(tensorflowModelUri).getInputStream()))
                    .inputNames(Collections.singletonList(inputName))
                    .outputNames(Arrays.asList(outputName))
                    .sessionOptionsConfigProto(configProto)
                    .build();
        } catch (IOException e) {
            throw new IllegalStateException(String.format("Failed to load TF model [%s] and input [%s]:", tensorflowModelUri, inputName), e);
        }
    }

    FaceAnnotation[] detectFace(Mat image) throws Exception {
        double m = 12D / minFaceSize;
        double minLayer = Math.min(image.rows(), image.cols()) * m;

        List<Double> scales = Utils.computeScalePyramid(m, minLayer, scaleFactor);

        Object[] stageOneResult = stage1(image, scales);

        Object[] stageTwoResult = stage2(image, (INDArray) stageOneResult[0], (StageState) stageOneResult[1]);

        INDArray[] stageThreeResult = stage3(image, (INDArray) stageTwoResult[0], (StageState) stageTwoResult[1]);

        INDArray totalBoxes = stageThreeResult[0];
        INDArray points = stageThreeResult[1];

        FaceAnnotation[] faceAnnotation = toFaceAnnotation(totalBoxes, points);

        return faceAnnotation;
    }

    Object[] stage1(Mat image, List<Double> scales) throws IOException {
        INDArray totalBoxes = Nd4j.empty();

        for (Double scale : scales) {
            Mat scaledImage = Utils.scaleImage(image, scale);

            INDArray img = loader.asMatrix(scaledImage);

            img = img.permute(0, 3, 2, 1);
            INDArray imgNormalized = img.sub(127.5).mul(0.0078125);

            Map<String, INDArray> resultMap = pNet.run(Collections.singletonMap("input_1:0", imgNormalized));
            INDArray out0 = resultMap.get("softmax_1/truediv:0");
            INDArray out1 = resultMap.get("conv2d_5/BiasAdd:0");

            INDArray boxes = Utils.generateBoundingBox(
                    out0.get(point(0), all(), all(), point(1)),
                    out1.get(point(0), all(), all(), all()), scale, stepsTreshold.get(0)
            )[0];

            if (!boxes.isEmpty()) {
//                System.out.println(scale);
//                System.out.println(boxes.shape()[0]);
                INDArray pick = Utils.nms(boxes.dup(), 0.5, nmsMethod.Union);
                if (boxes.length() > 0 && pick.length() > 0 && !pick.isEmpty()) {
                    boxes = boxes.get(indices(pick.toLongVector()), all());
                    if (totalBoxes.isEmpty()) {
                        totalBoxes = boxes;
                    }
                    else {
                        totalBoxes = Nd4j.concat(0, totalBoxes, boxes);
                    }
                }
            }
        }

        if (!totalBoxes.isEmpty()) {
            INDArray pick = Utils.nms(totalBoxes, 0.7, nmsMethod.Union);
            totalBoxes = totalBoxes.get(indices(pick.toLongVector()), all());

            INDArray x2 = totalBoxes.get(all(), point(2));
            INDArray x1 = totalBoxes.get(all(), point(0));
            INDArray y2 = totalBoxes.get(all(), point(3));
            INDArray y1 = totalBoxes.get(all(), point(1));

            INDArray regw = x2.sub(x1);
            INDArray regh = y2.sub(y1);

            INDArray qq1 = x1.add(totalBoxes.get(all(), point(5)).mul(regw));
            INDArray qq2 = y1.add(totalBoxes.get(all(), point(6)).mul(regh));
            INDArray qq3 = x2.add(totalBoxes.get(all(), point(7)).mul(regw));
            INDArray qq4 = y2.add(totalBoxes.get(all(), point(8)).mul(regh));

            totalBoxes = Nd4j.vstack(qq1, qq2, qq3, qq4).transpose();
            totalBoxes = Utils.rerec(totalBoxes.dup());

//            INDArray score = totalBoxes.get(all(), point(4));
//            totalBoxes = Nd4j.hstack(totalBoxes, Nd4j.expandDims(score, 1));

        }
        return new Object[] { totalBoxes, Utils.pad(totalBoxes, image.cols(), image.rows()) };
    }

    Object[] stage2(Mat image, INDArray totalBoxes, StageState stageState) throws IOException {
        int numBoxes = totalBoxes.isEmpty() ? 0 : (int) totalBoxes.shape()[0];

        if (numBoxes == 0) {
            return new Object[] { totalBoxes, stageState };
        }
        INDArray img = loader.asMatrix(image);
        img = img.get(point(0),all(),all(), all()).permute(1,2,0);

        INDArray tempImg1 = computeTempImage(img, numBoxes, stageState, 24);

        Map<String, INDArray> resultMap = rNet.run(Collections.singletonMap("input_2:0", tempImg1));
        INDArray out0 = resultMap.get("softmax_2/Softmax:0");
        INDArray out1 = resultMap.get("dense_3/BiasAdd:0");

        INDArray score = out0.get(all(), point(1));

        INDArray ipass = Nd4j.where(score.match(1, Conditions.greaterThanOrEqual(stepsTreshold.get(1))),
                null, null)[0];

        if(ipass.length() == 0){
            return new Object[] { Nd4j.empty(), stageState };
        }

        INDArray boxes = totalBoxes.get(indices(ipass.toLongVector()),all()).dup();
        INDArray s = score.get(indices(ipass.toLongVector())).dup();
        totalBoxes = Nd4j.hstack(boxes, Nd4j.expandDims(s,1));

        INDArray mv = out1.get(indices(ipass.toLongVector()) ,all());

        if (!totalBoxes.isEmpty() && totalBoxes.shape()[0] > 0) {
            INDArray pick = Utils.nms(totalBoxes.dup(), 0.7, nmsMethod.Union);

            totalBoxes = totalBoxes.get(indices(pick.toLongVector()), all());

            totalBoxes = Utils.bbreg(totalBoxes, mv.get(indices(pick.toLongVector()), all()));

            totalBoxes = Utils.rerec(totalBoxes); // rerec include floor
        }

        stageState = Utils.pad(totalBoxes, image.cols(), image.rows());

        return new Object[] { totalBoxes, stageState };
    }

    INDArray[] stage3(Mat image, INDArray totalBoxes, StageState stageState) throws IOException {
        int numBoxes = totalBoxes.isEmpty() ? 0 : (int) totalBoxes.shape()[0];

        if (numBoxes == 0) {
            return new INDArray[] { totalBoxes, Nd4j.empty() };
        }

        INDArray img = loader.asMatrix(image);
        img = img.get(point(0),all(),all(), all()).permute(1,2,0);

        INDArray tempImg1 = computeTempImage(img, numBoxes, stageState, 48);

        Map<String, INDArray> resultMap = oNet.run(Collections.singletonMap("input_3:0", tempImg1));

        INDArray out0 = resultMap.get("softmax_3/Softmax:0");
        INDArray out1 = resultMap.get("dense_6/BiasAdd:0");
        INDArray out2 = resultMap.get("dense_7/BiasAdd:0");

        INDArray score = out0.get(all(), point(1));

        INDArray ipass = Nd4j.where(score.match(1, Conditions.greaterThanOrEqual(stepsTreshold.get(2))),
                null, null)[0];

        if(ipass.length() == 0){
            return new INDArray[] { Nd4j.empty(), Nd4j.empty() };
        }

        INDArray boxes = totalBoxes.get(indices(ipass.toLongVector()),all()).dup();
        INDArray s = score.get(indices(ipass.toLongVector())).dup();
        totalBoxes = Nd4j.hstack(boxes, Nd4j.expandDims(s,1));

        INDArray mv = out1.get(indices(ipass.toLongVector()) ,all());

        INDArray points = out2.get(indices(ipass.toLongVector()), all()).transpose();

        INDArray w = totalBoxes.get(all(), point(2)).sub(totalBoxes.get(all(), point(0))).addi(1);
        INDArray h = totalBoxes.get(all(), point(3)).sub(totalBoxes.get(all(), point(1))).addi(1);

        points.put(new INDArrayIndex[] { interval(0, 5), all() },
                Nd4j.repeat(w, 5)
                        .mul(points.get(interval(0, 5), all()))
                        .add(Nd4j.repeat(totalBoxes.get(all(), point(0)), 5))
                        .sub(1));

        points.put(new INDArrayIndex[] { interval(5, 10), all() },
                Nd4j.repeat(h, 5)
                        .mul(points.get(interval(5, 10), all()))
                        .add(Nd4j.repeat(totalBoxes.get(all(), point(1)), 5))
                        .sub(1));

        if (totalBoxes.shape()[0] > 0) {

            totalBoxes = Utils.bbreg(totalBoxes.dup(), mv);

            INDArray pick = Utils.nms(totalBoxes.dup(), 0.7, nmsMethod.Min);

            totalBoxes = totalBoxes.get(indices(pick.toLongVector()), all());

            points = points.get(all(), indices(pick.toLongVector())).transpose();
        }

        return new INDArray[] { totalBoxes, points };
    }

    INDArray computeTempImage(INDArray image, int numBoxes, StageState stageState, int size) throws IOException {

        INDArray tempImg = Nd4j.zeros(size, size, 3, numBoxes);

        Size newSize = new Size(size, size);

        for (int k = 0; k < numBoxes; k++) {
            INDArray tmp = Nd4j.zeros(stageState.getTmph().getInt(k), stageState.getTmpw().getInt(k), 3);

            tmp.put(new INDArrayIndex[] {
                            interval(stageState.getDy().getInt(k) - 1, stageState.getEdy().getInt(k)),
                            interval(stageState.getDx().getInt(k) - 1, stageState.getEdx().getInt(k)),
                            all() },
                    image.get(
                            interval(stageState.getY().getInt(k) - 1, stageState.getEy().getInt(k)),
                            interval(stageState.getX().getInt(k) - 1, stageState.getEx().getInt(k)),
                            all()));

            if ((tmp.shape()[0] > 0 && tmp.shape()[1] > 0) || (tmp.shape()[0] == 0 && tmp.shape()[1] == 0)) {

                INDArray resizedImage = resizeArray(tmp.permute(2, 0, 1).dup(), newSize)
                        .get(point(0), all(), all(), all()).permute(1, 2, 0).dup();

                tempImg.put(new INDArrayIndex[] { all(), all(), all(), point(k) }, resizedImage);
            }
            else {
                return Nd4j.empty();
            }
        }

        tempImg = tempImg.subi(127.5).muli(0.0078125);

        INDArray tempImg1 = tempImg.permutei(3, 1, 0, 2).dup();

        return tempImg1;
    }

    INDArray resizeArray(INDArray imageCHW, Size newSizeWH) throws IOException {
        Assert.isTrue(imageCHW.size(0) == 3, "Input image is expected to have the [3, W, H] dimensions");
        // Mat expects [C, H, W] dimensions
        Mat mat = loader.asMat(imageCHW);
        resize(mat, mat, newSizeWH, 0, 0, CV_INTER_AREA);
        //[0, W, H, 3]
        return loader.asMatrix(mat);
    }

//    static INDArray softmax(INDArray input, int dimension){
//        INDArray x = input.sub(Nd4j.expandDims(Nd4j.max(input, dimension),dimension));
//        INDArray y = Transforms.exp(x);
//        return y.div(Nd4j.expandDims(Nd4j.sum(y, dimension),dimension));
//    }

    FaceAnnotation[] toFaceAnnotation(INDArray totalBoxes, INDArray points) {

        if (totalBoxes.isEmpty()) {
            return new FaceAnnotation[0];
        }

        org.springframework.util.Assert.isTrue(totalBoxes.rows() == points.rows(), "Inconsistent number of boxes ("
                + totalBoxes.rows() + ") + and points (" + points.rows() + ")");

        FaceAnnotation[] faceAnnotations = new FaceAnnotation[totalBoxes.rows()];
        for (int i = 0; i < totalBoxes.rows(); i++) {
            FaceAnnotation faceAnnotation = new FaceAnnotation();

            faceAnnotation.setBoundingBox(FaceAnnotation.BoundingBox.of(totalBoxes.getInt(i, 0), // x
                    totalBoxes.getInt(i, 1), // y
                    totalBoxes.getInt(i, 2) - totalBoxes.getInt(i, 0), // w
                    totalBoxes.getInt(i, 3) - totalBoxes.getInt(i, 1))); //h

            faceAnnotation.setConfidence(totalBoxes.getDouble(i, 4));

            faceAnnotation.setLandmarks(new FaceAnnotation.Landmark[5]);
            faceAnnotation.getLandmarks()[0] = FaceAnnotation.Landmark.of(FaceAnnotation.Landmark.LandmarkType.LEFT_EYE,
                    FaceAnnotation.Landmark.Position.of(points.getInt(i, 0), points.getInt(i, 5)));
            faceAnnotation.getLandmarks()[1] = FaceAnnotation.Landmark.of(FaceAnnotation.Landmark.LandmarkType.RIGHT_EYE,
                    FaceAnnotation.Landmark.Position.of(points.getInt(i, 1), points.getInt(i, 6)));
            faceAnnotation.getLandmarks()[2] = FaceAnnotation.Landmark.of(FaceAnnotation.Landmark.LandmarkType.NOSE,
                    FaceAnnotation.Landmark.Position.of(points.getInt(i, 2), points.getInt(i, 7)));
            faceAnnotation.getLandmarks()[3] = FaceAnnotation.Landmark.of(FaceAnnotation.Landmark.LandmarkType.MOUTH_LEFT,
                    FaceAnnotation.Landmark.Position.of(points.getInt(i, 3), points.getInt(i, 8)));
            faceAnnotation.getLandmarks()[4] = FaceAnnotation.Landmark.of(FaceAnnotation.Landmark.LandmarkType.MOUTH_RIGHT,
                    FaceAnnotation.Landmark.Position.of(points.getInt(i, 4), points.getInt(i, 9)));

            faceAnnotations[i] = faceAnnotation;
        }

        return faceAnnotations;
    }
}
