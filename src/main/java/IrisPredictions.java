import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;

public class IrisPredictions {
    public static void main(String[] args) throws IOException {

        String[] labelsIris = {"Iris-setosa","Iris-versicolor","Iris-virginca"};

        System.out.println("********************************");
        System.out.println("loading du model");

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(new File("irisModel.zip"));

        System.out.println("********************************");
        System.out.println("Model predictions");

        INDArray inputData = Nd4j.create(new double[][]{
                {5.1,3.5,1.4,0.2}, {4.9,3.0,1.4,0.2},
                {6.7,3.1,4.4,1.4}, {5.6,3.0,1.4,0.2},
                {5.1,3.5,4.5,1.5}, {6.0,3.0,4.8,1.8},
                {6.9,3.1,5.4,2.1}
        });

        INDArray output = model.output(inputData);
        int[] classes = output.argMax(1).toIntVector();// argMax donne l'index du max // 1 => ligne par ligne

        System.out.println(output);
        for (int i = 0; i <classes.length ; i++) {
            System.out.println("classe : "+labelsIris[classes[i]]);
        }
    }
}
