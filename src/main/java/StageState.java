import org.nd4j.linalg.api.ndarray.INDArray;

public class StageState{

    private final INDArray dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph;

    public StageState(INDArray dy, INDArray edy, INDArray dx, INDArray edx, INDArray y, INDArray ey, INDArray x, INDArray ex, INDArray tmpw, INDArray tmph) {
        this.dy = dy;
        this.edy = edy;
        this.dx = dx;
        this.edx = edx;
        this.y = y;
        this.ey = ey;
        this.x = x;
        this.ex = ex;
        this.tmpw = tmpw;
        this.tmph = tmph;
    }

    public INDArray getDy() {
        return dy;
    }

    public INDArray getEdy() {
        return edy;
    }

    public INDArray getDx() {
        return dx;
    }

    public INDArray getEdx() {
        return edx;
    }

    public INDArray getY() {
        return y;
    }

    public INDArray getEy() {
        return ey;
    }

    public INDArray getX() {
        return x;
    }

    public INDArray getEx() {
        return ex;
    }

    public INDArray getTmpw() {
        return tmpw;
    }

    public INDArray getTmph() {
        return tmph;
    }

    //    public StageState(){}
}