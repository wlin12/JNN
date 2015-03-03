package jnn.functions.composite;

import java.util.ArrayList;

import jnn.features.FeatureVector;
import jnn.functions.DenseArrayToDenseArrayTransform;
import jnn.functions.DenseArrayToDenseTransform;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.neuron.DenseNeuronArray;
import jnn.training.TreeInference;
import util.PrintUtils;
import util.RandomUtils;

public class DeepRNN extends Layer implements DenseArrayToDenseArrayTransform, DenseArrayToDenseTransform{	
	public ArrayList<Integer> dims = new ArrayList<Integer>();

	public ArrayList<RNN> hiddenLayers = new ArrayList<RNN>();

	private static final String HIDDENSTATEKEY = "hidden";
	public boolean nonlinear = false;

	public DeepRNN(int input) {
		super();
		dims.add(input);
	}

	public DeepRNN addLayer(int outputDim, int stateDim){
		DeepRNN newLSTM = new DeepRNN(dims.get(0));
		for(int i = 0; i < hiddenLayers.size(); i++){
			newLSTM.dims.add(dims.get(i+1));
			newLSTM.hiddenLayers.add(hiddenLayers.get(i));
		}
		newLSTM.dims.add(outputDim);
		newLSTM.hiddenLayers.add(new LongShortTermMemoryRNNVariableState(dims.get(dims.size()-1), stateDim, outputDim));
		return newLSTM;
	}
	
	public DeepRNN addLayer(int outputDim, int stateDim, String type, boolean forward, boolean backward, int outputSigmoid){
		DeepRNN newLSTM = new DeepRNN(dims.get(0));
		for(int i = 0; i < hiddenLayers.size(); i++){
			newLSTM.dims.add(dims.get(i+1));
			newLSTM.hiddenLayers.add(hiddenLayers.get(i));
		}
		newLSTM.dims.add(outputDim);
		RNN rnn = null;
		if(type.equals("rnn")){
			rnn = new RNN(dims.get(dims.size()-1), stateDim, outputDim);
		}
		else if(type.equals("lstm")){
			rnn = new LongShortTermMemoryRNNVariableState(dims.get(dims.size()-1), stateDim, outputDim);
		}
		else if(type.equals("convolution")){
			rnn = new ConvolutionLayer(2, dims.get(dims.size()-1), outputDim);
		}
		else{
			throw new RuntimeException("unknown rnn type");
		}
		if(forward){
			rnn.type_forward = 2;
		}
		else{			
			rnn.type_forward = 0;
		}
		if(backward){
			rnn.type_backward = 2;
		}
		else{			
			rnn.type_backward = 0;
		}
		rnn.outputSigmoid = outputSigmoid;
		newLSTM.hiddenLayers.add(rnn);
		return newLSTM;
	}
	
	public DeepRNN addLayer(int outputDim, int stateDim, String type, int forward, int backward, int outputSigmoid){
		DeepRNN newLSTM = new DeepRNN(dims.get(0));
		for(int i = 0; i < hiddenLayers.size(); i++){
			newLSTM.dims.add(dims.get(i+1));
			newLSTM.hiddenLayers.add(hiddenLayers.get(i));
		}
		newLSTM.dims.add(outputDim);
		RNN rnn = null;
		if(type.equals("rnn")){
			rnn = new RNN(dims.get(dims.size()-1), stateDim, outputDim);
		}
		else if(type.equals("lstm")){
			rnn = new LongShortTermMemoryRNNVariableState(dims.get(dims.size()-1), stateDim, outputDim);
		}
		else if(type.equals("convolution")){
			rnn = new ConvolutionLayer(2, dims.get(dims.size()-1), outputDim);
		}
		else{
			throw new RuntimeException("unknown rnn type");
		}
		rnn.type_forward = forward;
		rnn.type_backward = backward;
		rnn.outputSigmoid = outputSigmoid;
		newLSTM.hiddenLayers.add(rnn);
		return newLSTM;
	}

	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray[] output, int outputStart, int outputEnd,
			OutputMappingDenseArrayToDenseArray mapping) {
		TreeInference inference = mapping.getSubInference();

		// add input units
		for(int i = 0; i < input.length; i++){
			inference.addNeurons(0, input[i]);
		}
		DenseNeuronArray[][] hiddenLayerNeurons = new DenseNeuronArray[hiddenLayers.size()][input.length];
		DenseNeuronArray[] prevLayer = input;
		for(int d = 0; d < hiddenLayers.size(); d++){			
			for(int i = 0; i < input.length; i++){
				hiddenLayerNeurons[d][i] = new DenseNeuronArray(dims.get(d+1));
				inference.addNeurons(d+1, hiddenLayerNeurons[d][i]);
			}
			RNN transform = null;
			transform = hiddenLayers.get(d);
			if(d == hiddenLayers.size()-1 && !nonlinear){
				transform.outputSigmoid = 0;
			}
			else{
				transform.outputSigmoid = 2;				
			}
			inference.addMapping(new OutputMappingDenseArrayToDenseArray(prevLayer, hiddenLayerNeurons[d], transform));
			prevLayer = hiddenLayerNeurons[d];
		}
		mapping.setForwardParam(HIDDENSTATEKEY, hiddenLayerNeurons);

		inference.init();
		inference.forward();

		for(int i = 0; i < output.length; i++){
			DenseNeuronArray outputI = output[i];
			DenseNeuronArray hiddenI = hiddenLayerNeurons[hiddenLayers.size()-1][i];
			for(int d = 0; d < dims.get(dims.size()-1); d++){				
				outputI.addNeuron(d+outputStart, hiddenI.getNeuron(d));
			}			
		}
	}

	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingDenseArrayToDense mapping) {
		TreeInference inference = mapping.getSubInference();

		// add input units
		for(int i = 0; i < input.length; i++){
			inference.addNeurons(0, input[i]);
		}
		DenseNeuronArray[][] hiddenLayerNeurons = new DenseNeuronArray[hiddenLayers.size()-1][input.length];
		DenseNeuronArray[] prevLayer = input;
		for(int d = 0; d < hiddenLayers.size()-1; d++){			
			for(int i = 0; i < input.length; i++){
				hiddenLayerNeurons[d][i] = new DenseNeuronArray(dims.get(d+1));
				inference.addNeurons(d+1, hiddenLayerNeurons[d][i]);
			}
			RNN transform = null;
			transform = hiddenLayers.get(d);
			if(d == hiddenLayers.size()-1 && !nonlinear){
				transform.outputSigmoid = 0;
			}
			else{
				transform.outputSigmoid = 2;				
			}
			inference.addMapping(new OutputMappingDenseArrayToDenseArray(prevLayer, hiddenLayerNeurons[d], transform));
			prevLayer = hiddenLayerNeurons[d];
		}
		DenseNeuronArray outputNeurons = new DenseNeuronArray(dims.get(dims.size()-1));
		inference.addNeurons(inference.getMaxLevel()+1, outputNeurons);
		inference.addMapping(new OutputMappingDenseArrayToDense(prevLayer, outputNeurons, hiddenLayers.get(hiddenLayers.size()-1)));

		mapping.setForwardParam(HIDDENSTATEKEY, outputNeurons);

		inference.init();
		inference.forward();

		for(int d = 0; d < dims.get(dims.size()-1); d++){				
			output.addNeuron(d+outputStart, outputNeurons.getNeuron(d));
		}		
		
	}

	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, DenseNeuronArray[] output, int outputStart,
			int outputEnd, OutputMappingDenseArrayToDenseArray mapping) {
		TreeInference inference = mapping.getSubInference();
		DenseNeuronArray[][] hiddenLayerNeurons = (DenseNeuronArray[][])mapping.getForwardParam(HIDDENSTATEKEY);
		for(int i = 0; i < output.length; i++){
			DenseNeuronArray outputI = output[i];
			DenseNeuronArray hiddenI = hiddenLayerNeurons[hiddenLayers.size()-1][i];
			for(int d = 0; d < dims.get(dims.size()-1); d++){				
				hiddenI.addError(d, outputI.getError(d+outputStart));
			}
		}
		inference.backward();
	}

	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, DenseNeuronArray output, int outputStart,
			int outputEnd, OutputMappingDenseArrayToDense mapping) {
		TreeInference inference = mapping.getSubInference();
		DenseNeuronArray outputNeurons = (DenseNeuronArray)mapping.getForwardParam(HIDDENSTATEKEY);
		for(int d = 0; d < dims.get(dims.size()-1); d++){				
			outputNeurons.addError(d, output.getError(d+outputStart));
		}
		inference.backward();
	}

	@Override
	public void updateWeights(double learningRate, double momentum) {
		for(int i = 0; i < hiddenLayers.size(); i++){
			hiddenLayers.get(i).updateWeights(learningRate, momentum);			
		}
	}

	@Override
	public String toString() {
		String dimStr = "";
		for(int d = 0; d < dims.size(); d++){
			dimStr += dims.get(d) + "x";
		}
		String ret = "this is a deep lstm with dim = " + dimStr + "\n";
		for(int i = 0; i < hiddenLayers.size(); i++){
			ret += "\n hidden layer " + i + "\n" + hiddenLayers.get(i) + "\n";
		}
		return ret;
	}
	
	public int getOutputSize() {
		return dims.get(dims.size()-1);
	}

	public static void main(String[] args){
		int stateDim = 100;
		int inputDim = 50;
		int instances = 1000;
		double learningRate = 0.1;
		FeatureVector.useAdagradDefault = true;
		FeatureVector.commitMethodDefault = 0;
		DeepRNN recNN = new DeepRNN(inputDim);
		recNN = recNN.addLayer(stateDim, stateDim);
		recNN = recNN.addLayer(stateDim, stateDim);
		recNN = recNN.addLayer(stateDim, stateDim);

		int len = 4;
		double[][][] input = new double[instances][][];
		double[][][] expected = new double[instances][][];
		double[][] expectedFinal = new double[instances][];
		for(int i = 0; i < input.length; i++){
			input[i] = new double[len][];
			expected[i] = new double[len][];
			expectedFinal[i] = new double[stateDim];
			for(int j = 0; j < len; j++){
				input[i][j] = new double[inputDim];
				RandomUtils.initializeRandomArray(input[i][j], 0, 1);
				expected[i][j] = new double[stateDim];
				RandomUtils.initializeRandomArray(expected[i][j], 0, 1);
			}
			RandomUtils.initializeRandomArray(expectedFinal[i],0,1);
		}

		long startTime = System.currentTimeMillis();
		for(int iteration = 0; iteration < 1000; iteration++){
			long iterationStart = System.currentTimeMillis();
			int i = iteration % instances;
			TreeInference inference = new TreeInference(0);
			DenseNeuronArray[] inputNeurons = new DenseNeuronArray[len];
			DenseNeuronArray[] states = new DenseNeuronArray[len];
			for(int j = 0; j < len; j++){
				inputNeurons[j] = new DenseNeuronArray(inputDim);
				states[j] = new DenseNeuronArray(stateDim);
				inference.addNeurons(0, inputNeurons[j]);
				inference.addNeurons(1, states[j]);
			}
			DenseNeuronArray finalState = new DenseNeuronArray(stateDim);
			inference.addNeurons(1, finalState);
			inference.addMapping(new OutputMappingDenseArrayToDenseArray(inputNeurons, states, recNN));
			inference.addMapping(new OutputMappingDenseArrayToDense(inputNeurons, finalState, recNN));
			inference.init();
			for(int j = 0; j < len; j++){
				inputNeurons[j].init();
				inputNeurons[j].loadFromArray(input[i][j]);
			}
			inference.forward();
			double error = 0;
//			for(int j = 0; j < len; j++){				
//				states[j].computeErrorTan(expected[i][j]);
//				error += states[j].sqError();
//			}
			error /= len;
			
			finalState.computeErrorTan(expectedFinal[i]);
			error += finalState.sqError();
			
			inference.backward();			
			//			System.err.println("initial state");
			//			recNN.initialLayer.printWeights();
			//			System.err.println("transformation layer");
			//			recNN.transformLayer.printWeights();
			//			
			inference.commit(0);
			//inference.printNeurons();
			//			recNN.printNeurons();

			PrintUtils.printDoubleArray("output = ", states[len-1].copyAsArray(), false);
			PrintUtils.printDoubleArray("error = ", states[len-1].copyErrorAsArray(), false);
			PrintUtils.printDoubleArray("output final = ", finalState.copyAsArray(), false);
			PrintUtils.printDoubleArray("error final = ", finalState.copyErrorAsArray(), false);
			System.err.println("error " + error);

			double avgTime = (System.currentTimeMillis() - startTime)/(iteration+1);
			System.err.println("avg time " + avgTime);
			System.err.println("this iteration " + (System.currentTimeMillis() - iterationStart));
			
		}
	}

}
