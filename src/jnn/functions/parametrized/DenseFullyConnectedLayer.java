package jnn.functions.parametrized;

import java.io.BufferedReader;
import java.io.PrintStream;

import jnn.features.DenseFeatureMatrix;
import jnn.features.DenseFeatureVector;
import jnn.functions.DenseArrayToDenseArrayTransform;
import jnn.functions.DenseToDenseTransform;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.training.GlobalParameters;
import jnn.training.GraphInference;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import util.IOUtils;

public class DenseFullyConnectedLayer extends Layer implements DenseToDenseTransform, DenseArrayToDenseArrayTransform{
	public int inputDim;
	public int outputDim;
	public DenseFeatureMatrix weights;
	public DenseFeatureVector bias;
	public boolean addNoise = GlobalParameters.addNoiseDefault;
	public boolean useBias = true;

	public DenseFullyConnectedLayer(int inputDim, int outputDim) {
		this.inputDim = inputDim;
		this.outputDim = outputDim;
		weights = new DenseFeatureMatrix(inputDim, outputDim);		
		bias = new DenseFeatureVector(outputDim);
		if(inputDim != 0){
			initialize(false, true);
		}
	}

	public DenseFullyConnectedLayer(int inputDim, int outputDim, boolean useAdagrad, boolean useMomentum, boolean useAdadelta) {
		this.inputDim = inputDim;
		this.outputDim = outputDim;
		weights = new DenseFeatureMatrix(inputDim, outputDim, useAdagrad, useMomentum, useAdadelta);		
		bias = new DenseFeatureVector(outputDim, useAdagrad, useMomentum, useAdadelta);
		if(inputDim != 0){
			initialize(false, true);
		}
	}

	public void initialize(boolean sigmoid, boolean tanh){
		if(GlobalParameters.initializationType == 0){
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
		weights.normalizedInitializationHtan(inputDim+1, outputDim);
		bias.normalizedInitializationHtan(inputDim+1, outputDim);		
	}
	
	public void initializeForTanhSigmoid(int inputDim){
		weights.normalizedInitializationHtan(inputDim+1, outputDim);
		bias.normalizedInitializationHtan(inputDim+1, outputDim);		
	}

	public void initializeForLogisticSigmoid(){
		weights.normalizedInitializationSigmoid(inputDim+1, outputDim);
		bias.normalizedInitializationSigmoid(inputDim+1, outputDim);		
	}

	public void initializeUniform(){
		weights.initializeUniform(-0.1d,0.1d);
		bias.initializeUniform(-0.1d,0.1d);
	}

	public void propagate(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, Mapping mapping){
		INDArray W = weights.getWeights();
		INDArray x = input.getOutputRange(inputStart, inputEnd);
		INDArray yBias = bias.getWeights();		
		if(mapping.isTrain() && addNoise){
			W = W.add(weights.genGaussianNoise(mapping.getId()));
		}

		INDArray y = x.mmul(W);

		if(useBias){
			if(mapping.isTrain() && addNoise){
				yBias.addi(bias.genGaussianNoise(mapping.getId()));
			}
			y.addi(yBias);
		}
		output.setOutputRange(outputStart, outputEnd, y);	
	}

	public void backpropagate(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, Mapping mapping){
		INDArray WTranspose = weights.getTranspose();
		INDArray x = input.getOutputRange(inputStart, inputEnd);

		INDArray yGrad = output.getErrorRange(outputStart, outputEnd);

		INDArray xGrad = yGrad.mmul(WTranspose);
//		INDArray WGrad = x.transpose().mmul(yGrad);

		input.setErrorRange(inputStart, inputEnd, xGrad);

//		weights.storeGradients(mapping.getId(), WGrad);
		if(useBias){
			bias.storeGradients(mapping.getId(), yGrad.dup());
		}
		weights.storeInputsAndOutputs(mapping.getId(), x, yGrad);
	}

	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		propagate(input, inputStart, inputEnd, output, outputStart, outputEnd, mapping);
	}

	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray[] output, int outputStart, int outputEnd,
			OutputMappingDenseArrayToDenseArray mapping) {
		for(int i = 0; i < input.length; i++){
			propagate(input[i], inputStart, inputEnd, output[i], outputStart, outputEnd, mapping);			
		}
	}

	@Override
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd, OutputMappingDenseToDense mapping) {
		backpropagate(input, inputStart, inputEnd, output, outputStart, outputEnd, mapping);
	}

	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, DenseNeuronArray[] output, int outputStart,
			int outputEnd, OutputMappingDenseArrayToDenseArray mapping) {
		for(int i = 0; i < input.length; i++){
			backpropagate(input[i], inputStart, inputEnd, output[i], outputStart, outputEnd, mapping);			
		}
	}

	@Override
	public void updateWeights(double learningRate, double momentum) {
		if(useBias){
			bias.update();
		}
		weights.update();
	}

	public void save(PrintStream out){
		out.println(inputDim);
		out.println(outputDim);
		out.println(addNoise);
		out.println(useBias);		
		weights.save(out);
		bias.save(out);
	}

	public static DenseFullyConnectedLayer load(BufferedReader in) {
		try {
			int inputDim = Integer.parseInt(in.readLine());
			int outputDim = Integer.parseInt(in.readLine());
			DenseFullyConnectedLayer layer = new DenseFullyConnectedLayer(inputDim, outputDim);
			layer.addNoise = Boolean.parseBoolean(in.readLine());
			layer.useBias = Boolean.parseBoolean(in.readLine());
			layer.weights = DenseFeatureMatrix.load(in);
			layer.bias = DenseFeatureVector.load(in);
			return layer;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}
	
	@Override
	public void backwardEnd(GraphInference inference) {
		weights.checkinGradients(inference.getId());
	}

	@Override
	public String toString() {
		return "this is a " + inputDim + "x" + outputDim+" fully connected layer" + "\n weights:\n" + weights.getWeights() + "\n bias:\n" + bias.getWeights();
	}

	public int getOutputDim() {
		return outputDim;
	}

	public int getInputDim() {
		return inputDim;
	}

	public void checkinGradient(int processId){
		weights.checkinGradients(processId);
	}

	public static void main(String[] args){
		testForwardBackward();
	}

	public static void testSaveLoad(){
		DenseFullyConnectedLayer connectedLayer = new DenseFullyConnectedLayer(3, 5);
		double[][] weights = new double[][]{{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}};
		double[] bias = new double[]{1,2,3,4,5};
		connectedLayer.weights.initialize(weights);
		connectedLayer.bias.initialize(bias);
		connectedLayer.save(IOUtils.getPrintStream("/tmp/file"));

		System.err.println("original parameters");
		connectedLayer.save(System.err);

		DenseFullyConnectedLayer loadedLayer = DenseFullyConnectedLayer.load(IOUtils.getReader("/tmp/file"));
		System.err.println("loaded parameters");
		loadedLayer.save(System.err);
	}

	public static void testForwardBackward(){
		DenseFullyConnectedLayer connectedLayer = new DenseFullyConnectedLayer(3, 5);
		double[][] weights = new double[][]{{1,2,3,4,5}, {6,7,8,9,10}, {11,12,13,14,15}};
		double[] bias = new double[]{1,2,3,4,5};
		connectedLayer.weights.initialize(weights);
		connectedLayer.bias.initialize(bias);
		for(int e = 0; e < 4; e++){

			GraphInference inference = new GraphInference(0, true);
			DenseNeuronArray input = new DenseNeuronArray(3);
			DenseNeuronArray output = new DenseNeuronArray(5);
			input.init();
			input.addNeuron(0, 2);
			input.addNeuron(1, 1);
			input.addNeuron(2, 3);
			inference.addNeurons(0,input);
			inference.addNeurons(1,output);
			inference.addMapping(new OutputMappingDenseToDense(input, output, connectedLayer));
			inference.init();
			inference.forward();

			output.addError(0, 0.1);
			output.addError(1, 0.2);
			output.addError(2, 0.3);
			output.addError(3, 0.4);
			output.addError(4, 0.5);

			inference.backward();

			//		System.err.println(input);
			//		System.err.println(output);
		}
		connectedLayer.checkinGradient(0);
		connectedLayer.updateWeightsTimed(0, 0);
		connectedLayer.save(System.err);
	}

	public static void testForwardBackwardLarge(){
		DenseFullyConnectedLayer connectedLayer = new DenseFullyConnectedLayer(200, 150);
		for(int e = 0; e < 8000; e++){

			GraphInference inference = new GraphInference(0, true);
			DenseNeuronArray input = new DenseNeuronArray(200);
			DenseNeuronArray output = new DenseNeuronArray(150);
			input.init();
			input.addNeuron(0, 2);
			input.addNeuron(1, 1);
			input.addNeuron(2, 3);
			inference.addNeurons(0,input);
			inference.addNeurons(1,output);
			inference.addMapping(new OutputMappingDenseToDense(input, output, connectedLayer));
			inference.init();
			inference.forward();

			output.addError(0, 0.1);
			output.addError(1, 0.2);
			output.addError(2, 0.3);
			output.addError(3, 0.4);
			output.addError(4, 0.5);

			inference.backward();

			//		System.err.println(input);
			//		System.err.println(output);
		}
		connectedLayer.updateWeightsTimed(0, 0);
		connectedLayer.printCommitTimeAndReset();
		connectedLayer.save(System.err);
	}


	public void setL2(double l2){
		bias.setL2(l2);
		weights.setL2(l2);
	}

	public void normalizeWeights(){
		weights.normalize();

	}
}