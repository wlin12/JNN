package jnn.functions.parametrized;

import java.io.BufferedReader;
import java.io.PrintStream;
import java.util.HashSet;
import java.util.Map.Entry;

import jnn.features.DenseRowFeatureMatrix;
import jnn.features.FeatureVector;
import jnn.functions.SparseToDenseTransform;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingSparseToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.SparseNeuronArray;
import jnn.training.TreeInference;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import util.IOUtils;
import util.RandomUtils;

public class SparseFullyConnectedLayer extends Layer implements SparseToDenseTransform{

	int inputDim;
	int outputDim;

	DenseRowFeatureMatrix weights;

	public SparseFullyConnectedLayer(int inputDim, int outputDim) {
		super();
		this.inputDim = inputDim;
		this.outputDim = outputDim;
		weights = new DenseRowFeatureMatrix(inputDim, outputDim);
		initialize(false, true);
	}

	public void initialize(double[][] vals){
		weights.initialize(vals);
	}

	public void initialize(boolean sigmoid, boolean tanh){
		if(FeatureVector.initializationType == 0){
			initializeUniform();
		}
		else {
			if(sigmoid){
				initializeForLogisticSigmoid();
			}
			else if(tanh){
				initializeForTanhSigmoid();
			}
			else{
				initializeUniform();
			}
		}
	}

	public void initializeForTanhSigmoid(){
		weights.normalizedInitializationHtan(outputDim);
	}

	public void initializeForLogisticSigmoid(){
		weights.normalizedInitializationSigmoid(outputDim);
	}

	public void initializeUniform(){
		weights.initializeUniform(-0.1,0.1);
	}

	@Override
	public void forward(SparseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingSparseToDense mapping) {
		INDArray out = Nd4j.zeros(outputDim);
		for(Entry<Integer, Double> entry : input.getNonZeroEntries()){
			out.addi(weights.getUpdatedWeights(entry.getKey()));
			out.muli(entry.getValue());
		}
		output.setOutputRange(outputStart, outputEnd, out);
	}

	@Override
	public void backward(SparseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingSparseToDense mapping) {
		INDArray yGrad = output.getErrorRange(outputStart, outputEnd);
		for(Entry<Integer, Double> entry : input.getNonZeroEntries()){
			INDArray wGrad = yGrad.mul(entry.getValue());
			weights.storeGradient(entry.getKey(), mapping.getId(), wGrad);
		}
	}

	@Override
	public void updateWeights(double learningRate, double momentum) {
		weights.update();
	}
	
	public void updateWeights(HashSet<Integer> keysToUpdate) {
		weights.update(keysToUpdate);
	}
	
	public void save(PrintStream out){
		out.println(inputDim);
		out.println(outputDim);
		weights.save(out);
	}
	
	public static SparseFullyConnectedLayer load(BufferedReader in){
		try {
			int inputDim = Integer.parseInt(in.readLine());
			int outputDim = Integer.parseInt(in.readLine());
			SparseFullyConnectedLayer layer = new SparseFullyConnectedLayer(inputDim, outputDim);
			layer.weights = DenseRowFeatureMatrix.load(in);	
			return layer;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}



	public void setL2Reg(double projectionL2) {
		weights.setL2Reg(projectionL2);
	}

	public void setRegularizeWeights(int key, double[] embeddings) {
		weights.initialize(key, embeddings);
	}
	
	public void setRegularizeWeights(int key, double[] embeddings, boolean addRandomNoise) {
		if(addRandomNoise){
			for(int i = 0; i < embeddings.length; i++){
				embeddings[i]+=RandomUtils.initializeRandomNumber(-1, 1, 10)*embeddings[i];
			}
		}
		weights.initialize(key, embeddings);
	}

	public int getMax(int row) {
		return weights.getMax(row);
	}

	@Override
	public String toString() {
		return "this is a " + inputDim + "x" + outputDim +" sparse fully connected layer";
	}

	public INDArray getWeight(int row) {		
		return weights.getUpdatedWeights(row);
	}
	
	public static void main(String[] args){
		testSaveLoad();
	}
	
	public static void test1(){
		FeatureVector.useAdadeltaDefault = true;
		FeatureVector.l2regularizerLambdaDefault = 0.000;
		TreeInference inference = new TreeInference(0);

		SparseFullyConnectedLayer layer = new SparseFullyConnectedLayer(3, 5);
		double[][] weights = new double[][]{{1,2,3,4,5},{6,7,8,9,10}, {11,12,13,14,15}};
		layer.initialize(weights);

		SparseNeuronArray input = new SparseNeuronArray(3);
		input.init();
		input.setNeuron(0, 1);

		DenseNeuronArray input2 = new DenseNeuronArray(3);
		input2.init();
		input2.addNeuron(0, 1);

		DenseNeuronArray output = new DenseNeuronArray(5);
		output.init();

		OutputMappingSparseToDense map = new OutputMappingSparseToDense(0, 2, 0, 4, input, output, layer);
		map.setParentInference(inference);
		layer.forward(input, 0, 2, output, 0, 4, map);

		output.addError(0, 1);
		output.addError(1, 2);
		output.addError(2, 3);
		output.addError(3, 4);
		output.addError(4, 5);

		layer.backward(input, 0, 2, output, 0, 4, map);

		layer.updateWeights(0.1, 0);

		System.err.println("output for sparse layer");
		System.err.println(output);

		System.err.println("updated weights for sparse layer");
		for(int i = 0; i < 3; i++){
			System.err.println(layer.weights.getUpdatedWeights(i));
		}

		DenseFullyConnectedLayer layerDense = new DenseFullyConnectedLayer(3, 5);
		layerDense.weights.initialize(weights);
		output.init();

		OutputMappingDenseToDense map2 = new OutputMappingDenseToDense(0, 2, 0, 4, input2, output, layerDense);
		map2.setParentInference(inference);
		layerDense.forward(input2, 0, 2, output, 0, 4, map2);

		output.addError(0, 1);
		output.addError(1, 2);
		output.addError(2, 3);
		output.addError(3, 4);
		output.addError(4, 5);

		layerDense.backward(input2, 0, 2, output, 0, 4, map2);
		layerDense.updateWeights(0.1, 0);
		System.err.println("updated weights for dense layer");
		System.err.println(layerDense.weights.getWeights());

	}
	
	public static void testSaveLoad(){
		SparseFullyConnectedLayer layer = new SparseFullyConnectedLayer(3, 5);
		double[][] weights = new double[][]{{1,2,3,4,5},{6,7,8,9,10}, {11,12,13,14,15}};
		layer.initialize(weights);
		
		layer.save(IOUtils.getPrintStream("/tmp/file"));
		System.err.println("original layer");
		layer.save(System.err);
		
		SparseFullyConnectedLayer layerLoaded = SparseFullyConnectedLayer.load(IOUtils.getReader("/tmp/file"));
		System.err.println("loaded layer");		
		layerLoaded.save(System.err);
	}
}
