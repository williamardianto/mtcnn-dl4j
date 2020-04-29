import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.*;

public class Utils {

    static List<Double> computeScalePyramid(double m, double minLayer, double scaleFactor) {
        List<Double> scales = new ArrayList<>();
        int factorCount = 0;

        while (minLayer >= 12) {
            scales.add(m * Math.pow(scaleFactor, factorCount));
            minLayer = minLayer * scaleFactor;
            factorCount++;
        }
        return scales;
    }

    static Mat scaleImage(Mat image, double scale) {
        int widthScaled = (int) Math.ceil(image.cols() * scale);
        int heightScaled = (int) Math.ceil(image.rows() * scale);

        Mat resizedimage = new Mat();
        resize(image, resizedimage, new Size(widthScaled, heightScaled), 0, 0, INTER_AREA);

//        imshow("Input Image", resizedimage);
//        if (waitKey(0) == 27) {
//            destroyAllWindows();
//        }
//        im_data_normalized = (im_data - 127.5) * 0.0078125
        return resizedimage;
    }

    public static INDArray bbreg(INDArray boundingBox, INDArray reg) {

        if (reg.shape()[1] == 1) {
            reg = reg.transpose();
        }
        INDArray w = boundingBox.get(all(), point(2)).sub(boundingBox.get(all(), point(0))).addi(1);
        INDArray h = boundingBox.get(all(), point(3)).sub(boundingBox.get(all(), point(1))).addi(1);
        INDArray b1 = boundingBox.get(all(), point(0)).add(reg.get(all(), point(0)).mul(w));
        INDArray b2 = boundingBox.get(all(), point(1)).add(reg.get(all(), point(1)).mul(h));
        INDArray b3 = boundingBox.get(all(), point(2)).add(reg.get(all(), point(2)).mul(w));
        INDArray b4 = boundingBox.get(all(), point(3)).add(reg.get(all(), point(3)).mul(h));

        boundingBox.put(new INDArrayIndex[] { all(), interval(0, 4) }, Nd4j.vstack(b1, b2, b3, b4).transpose());
        return boundingBox;
    }



    public static INDArray rerec(INDArray bbox) {
        // convert bbox to square
        INDArray h = bbox.get(all(), point(3)).sub(bbox.get(all(), point(1)));
        INDArray w = bbox.get(all(), point(2)).sub(bbox.get(all(), point(0)));
        INDArray l = Transforms.max(w, h);

        bbox.put(new INDArrayIndex[] { all(), point(0) }, bbox.get(all(), point(0)).add(w.mul(0.5)).sub(l.mul(0.5)));
        bbox.put(new INDArrayIndex[] { all(), point(1) }, bbox.get(all(), point(1)).add(h.mul(0.5)).sub(l.mul(0.5)));
        INDArray lTile = Nd4j.repeat(l, 2).transpose();
        bbox.put(new INDArrayIndex[] { all(), interval(2, 4) }, bbox.get(all(), interval(0, 2)).add(lTile));

        bbox = Transforms.floor(bbox);

        return bbox;
    }

    static INDArray nms(INDArray boxes, double threshold, MTCNN.nmsMethod method){
        INDArray x1 = boxes.get(all(), point(0));
        INDArray y1 = boxes.get(all(), point(1));
        INDArray x2 = boxes.get(all(), point(2));
        INDArray y2 = boxes.get(all(), point(3));
        INDArray s = boxes.get(all(), point(4));

        INDArray area = (x2.sub(x1).add(1)).mul(y2.sub(y1).add(1));

        INDArray sortedS = Nd4j.sortWithIndices(s, 0, true)[0];

        INDArray pick = Nd4j.zerosLike(s);
        int counter = 0;

        while (sortedS.size(0) > 0) {
            if (sortedS.size(0) == 1) {
                pick.put(counter++, sortedS.dup());
                break;
            }
            long lastIndex = sortedS.size(0) - 1;
            INDArray i = sortedS.get(point(lastIndex));
            INDArray idx = sortedS.get(interval(0, lastIndex));
            pick.put(counter++, i.dup());

            INDArray xx1 = Transforms.max(x1.get(Nd4j.expandDims(idx,0)), x1.get(point(i.getInt(0))));
            INDArray yy1 = Transforms.max(y1.get(Nd4j.expandDims(idx,0)), y1.get(point(i.getInt(0))));
            INDArray xx2 = Transforms.min(x2.get(Nd4j.expandDims(idx,0)), x2.get(point(i.getInt(0))));
            INDArray yy2 = Transforms.min(y2.get(Nd4j.expandDims(idx,0)), y2.get(point(i.getInt(0))));

            INDArray w = Transforms.max(xx2.sub(xx1).add(1), 0.0f);
            INDArray h = Transforms.max(yy2.sub(yy1).add(1), 0.0f);
            INDArray inter = w.mul(h);

            int areaI = area.get(point(i.getInt(0))).getInt(0);
            INDArray o = (method == MTCNN.nmsMethod.Min) ?
                    inter.div(Transforms.min(area.get(Nd4j.expandDims(idx, 0)), areaI)):
                    inter.div(area.get(Nd4j.expandDims(idx, 0)).add(areaI).sub(inter));

            INDArray oIdx = Nd4j.where(
                    o.match(1, Conditions.lessThanOrEqual(threshold)), null, null)[0];

            if(oIdx.length() == 0){
                break;
            }

            sortedS = sortedS.get(Nd4j.expandDims(oIdx.castTo(DataType.FLOAT), 0));

        }

        return (counter == 0) ? Nd4j.empty() : pick.get(interval(0, counter));
    }

    static INDArray[] generateBoundingBox(INDArray imap, INDArray reg, double scale, double stepThreshold) {
        int stride = 2;
        int cellSize = 12;

        INDArray dx1 = reg.get(all(), all(), point(0));
        INDArray dy1 = reg.get(all(), all(), point(1));
        INDArray dx2 = reg.get(all(), all(), point(2));
        INDArray dy2 = reg.get(all(), all(), point(3));

        INDArray[] matchIndexes = Nd4j.where(imap.match(1, Conditions.greaterThanOrEqual(stepThreshold)), null, null);

        if(matchIndexes.length == 1){
            return new INDArray[] { Nd4j.empty(), Nd4j.empty() };
        }
        INDArray yx = Nd4j.vstack(matchIndexes[0], matchIndexes[1]).castTo(DataType.FLOAT);
        INDArray score = Nd4j.expandDims(imap.get(yx),1);

        reg = Nd4j.vstack(dx1.get(yx), dy1.get(yx), dx2.get(yx), dy2.get(yx)).transpose();

        INDArray bb = yx.transpose();
        INDArray q1 = Transforms.floor(bb.mul(stride).add(1).div(scale));
        INDArray q2 = Transforms.floor(bb.mul(stride).add(cellSize).div(scale));

        INDArray boundingBox = Nd4j.hstack(q1, q2, score, reg);

        return new INDArray[] { boundingBox, reg };
    }


    public static StageState pad(INDArray totalBoxes, int w, int h){
        INDArray tmpW = totalBoxes.get(all(), point(2)).sub(totalBoxes.get(all(), point(0))).add(1);
        INDArray tmpH = totalBoxes.get(all(), point(3)).sub(totalBoxes.get(all(), point(1))).add(1);
        long numBox = totalBoxes.shape()[0];

        INDArray dx = Nd4j.ones(numBox);
        INDArray dy = Nd4j.ones(numBox);
        INDArray edx = tmpW;
        INDArray edy = tmpH;

        INDArray x = Transforms.floor(totalBoxes.get(all(), point(0)));
        INDArray y = Transforms.floor(totalBoxes.get(all(), point(1)));
        INDArray ex = Transforms.floor(totalBoxes.get(all(), point(2)));
        INDArray ey = Transforms.floor(totalBoxes.get(all(), point(3)));

        INDArray tmp = Nd4j.where(ex.match(1, Conditions.greaterThan(w)), null, null)[0];
        if (tmp.length() != 0) {
            INDArray tmp2 = Nd4j.expandDims(tmp, 0);
            INDArray b = ex.get(tmp2).rsub(w).add(tmpW.get(tmp2));
            edx = edx.put(new INDArrayIndex[] { indices(tmp.toLongVector()) }, b);
            ex = ex.put(new INDArrayIndex[] { indices(tmp.toLongVector())}, Nd4j.ones(tmp.length()).mul(w));
        }

        tmp = Nd4j.where(ey.match(1, Conditions.greaterThan(h)), null, null)[0];
        if (tmp.length() != 0) {
            INDArray tmp2 = Nd4j.expandDims(tmp, 0);
            INDArray b = ey.get(tmp2).rsub(h).add(tmpH.get(tmp2));
            edy = edy.put(new INDArrayIndex[] { indices(tmp.toLongVector()) }, b);
            ey = ey.put(new INDArrayIndex[] { indices(tmp.toLongVector())}, Nd4j.ones(tmp.length()).mul(h));
        }

        tmp = Nd4j.where(x.match(1, Conditions.lessThan(1)), null, null)[0];
        if (tmp.length() != 0) {
            INDArray tmp2 = Nd4j.expandDims(tmp, 0);
            INDArray b = x.get(tmp2).rsub(2);

            dx.put(new INDArrayIndex[] { indices(tmp.toLongVector()) }, b);
            x = x.put(new INDArrayIndex[] { indices(tmp.toLongVector())}, Nd4j.ones(tmp.length()));
        }

        tmp = Nd4j.where(y.match(1, Conditions.lessThan(1)), null, null)[0];
        if (tmp.length() != 0) {
            INDArray tmp2 = Nd4j.expandDims(tmp, 0);
            INDArray b = y.get(tmp2).rsub(2);

            dy.put(new INDArrayIndex[] { indices(tmp.toLongVector()) }, b);
            y = y.put(new INDArrayIndex[] { indices(tmp.toLongVector())}, Nd4j.ones(tmp.length()));
        }

        return new StageState(dy, edy, dx, edx, y, ey, x, ex, tmpW, tmpH);
    }
}
