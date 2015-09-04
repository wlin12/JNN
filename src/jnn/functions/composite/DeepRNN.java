package jnn.functions.composite;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;

import jnn.functions.DenseArrayToDenseArrayTransform;
import jnn.functions.DenseArrayToDenseTransform;
import jnn.functions.composite.lstm.BLSTM;
import jnn.functions.composite.rnn.RNN;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.neuron.DenseNeuronArray;
import jnn.training.GlobalParameters;
import jnn.training.GraphInference;
import util.IOUtils;
import util.PrintUtils;
import util.RandomUtils;
import util.SerializeUtils;

public class DeepRNN extends Layer implements DenseArrayToDenseArrayTransform, DenseArrayToDenseTransform{	
	public ArrayList<Integer> dims = new ArrayList<Integer>();

	public ArrayList<RNN> hiddenLayers = new ArrayList<RNN>();
	public ArrayList<String> layerType = new ArrayList<String>();

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
			newLSTM.layerType.add(layerType.get(i));
		}
		newLSTM.dims.add(outputDim);
		newLSTM.hiddenLayers.add(new BLSTM(dims.get(dims.size()-1), stateDim, outputDim));
		newLSTM.layerType.add("lstm");
		return newLSTM;
	}

	public DeepRNN addLayer(int outputDim, int stateDim, String type, boolean forward, boolean backward, int outputSigmoid){
		DeepRNN newLSTM = new DeepRNN(dims.get(0));
		for(int i = 0; i < hiddenLayers.size(); i++){
			newLSTM.dims.add(dims.get(i+1));
			newLSTM.hiddenLayers.add(hiddenLayers.get(i));
			newLSTM.layerType.add(layerType.get(i));
		}
		newLSTM.dims.add(outputDim);
		RNN rnn = null;
		if(type.equals("rnn")){
			rnn = new RNN(dims.get(dims.size()-1), stateDim, outputDim);
		}
		else if(type.equals("lstm")){
			rnn = new BLSTM(dims.get(dims.size()-1), stateDim, outputDim);
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
		newLSTM.layerType.add(type);
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
			rnn = new BLSTM(dims.get(dims.size()-1), stateDim, outputDim);
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
		GraphInference inference = mapping.getSubInference();
		// add input units
		HashSet<DenseNeuronArray> existing = new HashSet<DenseNeuronArray>();
		for(int i = 0; i < input.length; i++){
			if(!existing.contains(input[i])){
				inference.addNeurons(0, input[i]);
				existing.add(input[i]);
			}
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
		GraphInference inference = mapping.getSubInference();
		// add input units
		HashSet<DenseNeuronArray> existing = new HashSet<DenseNeuronArray>();
		for(int i = 0; i < input.length; i++){
			if(!existing.contains(input[i])){
				inference.addNeurons(0, input[i]);
				existing.add(input[i]);
			}
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
		GraphInference inference = mapping.getSubInference();
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
		GraphInference inference = mapping.getSubInference();
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

	public void save(PrintStream out){
		int[] dimsArray = new int[dims.size()];
		for(int i = 0 ; i < dims.size(); i++){
			dimsArray[i] = dims.get(i);
		}
		SerializeUtils.saveIntArray(dimsArray,  out);
		for(String type : layerType){
			out.println(type);
		}
		Iterator<String> it = layerType.iterator();
		
		for(RNN rnn : hiddenLayers){
			String type = it.next();
			if(type.equals("rnn")){
				rnn.save(out);
			}
			else if(type.equals("lstm")){
				((BLSTM)rnn).save(out);
			}
			else if(type.equals("convolution")){
				((ConvolutionLayer)rnn).save(out);
			}
			else{
				throw new RuntimeException("unknown rnn type");
			}
		}		
		out.println(nonlinear);
	}

	public static DeepRNN load(BufferedReader in){
		int[] dims = SerializeUtils.loadIntArray(in);
		DeepRNN rnn = new DeepRNN(dims[0]);
		ArrayList<RNN> layers = new ArrayList<RNN>();
		ArrayList<Integer> dimArray = new ArrayList<Integer>();
		ArrayList<String> types = new ArrayList<String>();
		dimArray.add(dims[0]);
		try {
			for(int i = 0; i < dims.length-1; i++){
				types.add(in.readLine());
			}
			for(int i = 0; i < dims.length-1; i++){
				dimArray.add(dims[i+1]);
				if(types.get(i).equals("rnn")){
					layers.add(RNN.load(in));
				}
				else if(types.get(i).equals("lstm")){
					layers.add(BLSTM.load(in));
				}
				else if(types.get(i).equals("convolution")){
					layers.add(ConvolutionLayer.load(in));
				}
				else{
					throw new RuntimeException("unknown rnn type");
				}


				rnn.nonlinear = Boolean.parseBoolean(in.readLine());
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}

		rnn.dims = dimArray;
		rnn.hiddenLayers = layers;
		rnn.layerType = types;

		return rnn;
	}

	public static void main(String[] args){
		int stateDim = 100;
		int inputDim = 50;
		int instances = 1000;
		double learningRate = 0.1;
		GlobalParameters.useAdagradDefault = true;
		GlobalParameters.commitMethodDefault = 0;
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
		for(int iteration = 0; iteration < 2; iteration++){
			long iterationStart = System.currentTimeMillis();
			int i = 0 % instances;
			GraphInference inference = new GraphInference(0, true);
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

			//			inference.commit(0);

			PrintUtils.printDoubleArray("output = ", states[len-1].copyAsArray(), false);
			PrintUtils.printDoubleArray("error = ", states[len-1].copyErrorAsArray(), false);
			PrintUtils.printDoubleArray("output final = ", finalState.copyAsArray(), false);
			PrintUtils.printDoubleArray("error final = ", finalState.copyErrorAsArray(), false);
			System.err.println("error " + error);

			double avgTime = (System.currentTimeMillis() - startTime)/(iteration+1);
			System.err.println("avg time " + avgTime);
			System.err.println("this iteration " + (System.currentTimeMillis() - iterationStart));

			recNN.save(IOUtils.getPrintStream("/tmp/file"));
			recNN = DeepRNN.load(IOUtils.getReader("/tmp/file"));
		}
	}

}
