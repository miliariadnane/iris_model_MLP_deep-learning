import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class IrisApp {
    public static void main(String[] args) throws Exception {

        double learningRate=0.001;
        int numInputs=4;
        int numHidden=10; // nombre des neurons à utliser dans la couche caché
        int numOuputs=3;
        int batchSize=1;
        int classIndex=4; // l'emplacement de l'output dans 4ème position

        System.out.println("********************************");
        System.out.println("configuration du model");

        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(learningRate)) // les poids des neurons
                .weightInit(WeightInit.XAVIER) // INIT des poids de neurons => defaut is XAVIER
                .list()
                .layer(0,new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHidden)
                        .activation(Activation.SIGMOID).build())
                .layer(1,new OutputLayer.Builder()
                        .nIn(numHidden)
                        .nOut(numOuputs)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .activation(Activation.SOFTMAX).build())
                .build();

        // model
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

        // pour démmarage du serveur mootiring du processus d'apprentissage
        UIServer uiServer=UIServer.getInstance();
        InMemoryStatsStorage inMemoryStatsStorage=new InMemoryStatsStorage();
        // stocker les infos au cours d'apprentissage
        uiServer.attach(inMemoryStatsStorage);
        //model.setListeners(new ScoreIterationListener(10));
        model.setListeners(new StatsListener(inMemoryStatsStorage));


        ///////////////////////////// afficher la configuration
        //System.out.println(configuration.toJson());

        System.out.println("********************************");
        System.out.println("entrainement du model");

        File fileTrain = new ClassPathResource("iris-train.csv").getFile();
        RecordReader recordReaderTrain = new CSVRecordReader();
        recordReaderTrain.initialize(new FileSplit(fileTrain)); //split pour lire ligne par ligne
        DataSetIterator dataSetIteratorTrain =
                new RecordReaderDataSetIterator(recordReaderTrain, batchSize, classIndex, numOuputs);

        // AFFICHER DATASET
        /*
        while(dataSetIteratorTrain.hasNext()){
            System.out.println("-----------------------------");
            // parcourir dataset batch par batch
            // si batch = 1 càd qu'on entrain de lire dataset ligne par ligne
            DataSet dataSet = dataSetIteratorTrain.next();
            System.out.println(dataSet.getFeatures()); // features : INDArray
            System.out.println(dataSet.getLabels()); // les sorties => outputs
        }
        */
        int nEpochs=100;
        for (int i = 0; i <nEpochs ; i++) {
            model.fit(dataSetIteratorTrain);
        }

        System.out.println("********************************");
        System.out.println("Model Evaluation");

        File fileTest=new ClassPathResource("irisTest.csv").getFile();
        RecordReader recordReaderTest=new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(fileTest));
        DataSetIterator dataSetIteratorTest=
                new RecordReaderDataSetIterator(recordReaderTest,batchSize,classIndex,numOuputs);
        Evaluation evaluation=new Evaluation(numOuputs);

        while (dataSetIteratorTest.hasNext()){
            DataSet dataSet = dataSetIteratorTest.next();
            INDArray features=dataSet.getFeatures(); //inputs => de type INDArray
            INDArray labels=dataSet.getLabels(); //outputs
            INDArray predicted=model.output(features);
            evaluation.eval(labels,predicted); // comparer les entrées (inputs) avec resultats attendus
        }
        System.out.println(evaluation.stats());

        System.out.println("********************************");
        System.out.println("Enregistrement du model");

        ModelSerializer.writeModel(model,"irisModel.zip",true);


    }
}
