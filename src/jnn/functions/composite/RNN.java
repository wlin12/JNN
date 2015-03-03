package jnn.functions.composite;

import jnn.features.FeatureVector;
import jnn.functions.DenseArrayToDenseArrayTransform;
import jnn.functions.DenseArrayToDenseTransform;
import jnn.functions.nonparametrized.LogisticSigmoidLayer;
import jnn.functions.nonparametrized.TanSigmoidLayer;
import jnn.functions.parametrized.CopyLayer;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.functions.parametrized.StaticLayer;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.training.TreeInference;
import util.PrintUtils;
import util.RandomUtils;

public class RNN extends Layer implements DenseArrayToDenseArrayTransform, DenseArrayToDenseTransform{

	private class RNNParameters {
		DenseFullyConnectedLayer inputTransformLayer;
		StaticLayer initialStateLayer;
		
		public RNNParameters(int inputDim, int stateDim) {
			inputTransformLayer = new DenseFullyConnectedLayer(inputDim+stateDim, stateDim);
			inputTransformLayer.initialize(true, false);
			initialStateLayer = new StaticLayer(stateDim);
		}
		
		public void update(double learningRate, double momentum){
			inputTransformLayer.updateWeights(learningRate, momentum);
			initialStateLayer.updateWeights(learningRate, momentum);

		}
	}
	
	private class RNNBlock {
		DenseNeuronArray hprevState;

		DenseNeuronArray hState;

		int start;
		int end;

		public RNNBlock(DenseNeuronArray hprevState, int start) {
			super();
			this.hprevState = hprevState;
			this.start = start;
		}

		public RNNBlock nextState(){
			return new RNNBlock(hState, end+1);
		}

		public void addToInference(TreeInference inference, DenseNeuronArray inputX, RNNParameters parameters){
			int units = inputX.len();
			int stateSize = hprevState.size;

			int level = start;
			
			DenseNeuronArray stateInput = new DenseNeuronArray(units + stateSize);
			inference.addNeurons(level, stateInput);
			
			inference.addMapping(new OutputMappingDenseToDense(0,units-1,0,units-1,inputX, stateInput, CopyLayer.singleton));
			inference.addMapping(new OutputMappingDenseToDense(0,stateSize-1,units,units+stateSize-1,hprevState, stateInput, CopyLayer.singleton));
			level++;

			DenseNeuronArray stateOutput = new DenseNeuronArray(stateSize);
			inference.addNeurons(level, stateOutput);
			
			inference.addMapping(new OutputMappingDenseToDense(stateInput, stateOutput, parameters.inputTransformLayer));
			
			//next state computation
			level++;
			hState = new DenseNeuronArray(stateSize);
			hState.setName("state at level " + start);
			inference.addNeurons(level, hState);
			inference.addMapping(new OutputMappingDenseToDense(stateOutput , hState, LogisticSigmoidLayer.singleton));
		
			end = level+1;
		}
	}

	RNNParameters parameters;
	RNNParameters parametersBackward;

	//combiner Layer
	DenseFullyConnectedLayer combiner;
	
	int inputDim;
	int stateDim;
	int outputDim;

	public int type_forward = 2; // 0 -> none, 1 -> no_center_word, 2 -> with_center_word
	public int type_backward = 2; // 0 -> none, 1 -> no_center_word, 2 -> with_center_word
	
	public int outputSigmoid = 2; // 0 -> none, 1 -> sigmoid, 2 -> tanh;
	private static final String FORWARDKEY = "forwardblocks";
	private static final String BACKWARDKEY = "backwardblocks";
	private static final String COMBINEDKEY = "combinedblocks";
	private static final String COMBINEDFINALBLOCKKEY = "combinedfinalblock";

	public RNN(int inputDim, int stateDim, int outputDim) {
		this.inputDim = inputDim;
		this.stateDim = stateDim;
		this.outputDim = outputDim;
		
		parameters = new RNNParameters(inputDim, stateDim);
		parametersBackward = new RNNParameters(inputDim, stateDim);

		combiner = new DenseFullyConnectedLayer(stateDim*2, outputDim);
		combiner.initializeForTanhSigmoid();
		
	}
	
	public void buildInference(DenseNeuronArray[] input, int inputStart, int inputEnd, Mapping map){
		TreeInference inference = map.getSubInference();

		// add input units
		for(int i = 0; i < input.length; i++){
			inference.addNeurons(0, input[i]);
		}
		int startLevel = 2;

		if(type_forward != 0){
			DenseNeuronArray initialState = new DenseNeuronArray(stateDim);
			inference.addNeurons(1, initialState);
			
//			OutputMappingVoidToDense initialStateMapping = new OutputMappingVoidToDense(initialState, parameters.initialStateLayer);
//			inference.addMapping(0, 1, initialStateMapping);
//

			RNNBlock blockForward = new RNNBlock(initialState, startLevel);
			RNNBlock[] blocksForward = new RNNBlock[input.length];
			for(int i = 0; i < input.length; i++){
				blockForward.addToInference(inference, input[i], parameters);
				blocksForward[i] = blockForward;
				blockForward = blockForward.nextState();
			}
			map.setForwardParam(FORWARDKEY, blocksForward);
		}
		if(type_backward != 0){
			DenseNeuronArray initialState = new DenseNeuronArray(stateDim);
			inference.addNeurons(1, initialState);

//			OutputMappingVoidToDense initialStateMapping = new OutputMappingVoidToDense(initialState, parametersBackward.initialStateLayer);
//			inference.addMapping(0, 1, initialStateMapping);
//
//			OutputMappingVoidToDense initialCellMapping = new OutputMappingVoidToDense(initialCell, parametersBackward.initialCellLayer);
//			inference.addMapping(0, 1, initialCellMapping);

			RNNBlock blockBackward = new RNNBlock(initialState, startLevel);
			RNNBlock[] blocksBackward = new RNNBlock[input.length];
			for(int i = 0; i < input.length; i++){
				blockBackward.addToInference(inference, input[input.length-i-1], parametersBackward);
				blocksBackward[input.length-i-1] = blockBackward;
				blockBackward = blockBackward.nextState();
			}
			map.setForwardParam(BACKWARDKEY, blocksBackward);
		}
	}
	
	public void buildCombinerSequence(DenseNeuronArray[] input, int inputStart, int inputEnd, Mapping map){
		TreeInference inference = map.getSubInference();
		RNNBlock[] blocksForward = (RNNBlock[])map.getForwardParam(FORWARDKEY);
		RNNBlock[] blocksBackward = (RNNBlock[])map.getForwardParam(BACKWARDKEY);

		DenseNeuronArray[] blocksCombinedInput = new DenseNeuronArray[input.length];
		DenseNeuronArray[] blocksCombined = new DenseNeuronArray[input.length];
		int combinedLevel = inference.getMaxLevel()+1;
		for(int i = 0; i < input.length; i++){
			blocksCombinedInput[i] = new DenseNeuronArray(stateDim*2);
			blocksCombinedInput[i].setName("combined input " + i);
			inference.addNeurons(combinedLevel, blocksCombinedInput[i]);
			blocksCombined[i] = new DenseNeuronArray(outputDim);
			blocksCombined[i].setName("combined " + i);
			inference.addNeurons(combinedLevel+1, blocksCombined[i]);
			DenseNeuronArray forwardBlock = null;
			DenseNeuronArray backwardBlock = null;
			if(type_forward == 1){
				forwardBlock = blocksForward[i].hprevState;
			}
			if(type_forward == 2){
				forwardBlock = blocksForward[i].hState;
			}
			if(type_backward == 1){
				backwardBlock = blocksBackward[i].hprevState;
			}
			if(type_backward == 2){
				backwardBlock = blocksBackward[i].hState;
			}
			if(type_forward != 0){
				inference.addMapping(new OutputMappingDenseToDense(0, stateDim-1, 0, stateDim-1, forwardBlock, blocksCombinedInput[i], CopyLayer.singleton));
			}
			if(type_backward != 0){
				inference.addMapping(new OutputMappingDenseToDense(0, stateDim-1, stateDim, stateDim*2-1, backwardBlock, blocksCombinedInput[i], CopyLayer.singleton));
			}
			inference.addMapping(new OutputMappingDenseToDense(blocksCombinedInput[i], blocksCombined[i], combiner));
		}
		if(outputSigmoid == 0){
			map.setForwardParam(COMBINEDKEY, blocksCombined);
		}
		else if(outputSigmoid == 1){
			DenseNeuronArray[] blocksCombinedSig = new DenseNeuronArray[input.length];		
			for(int i = 0; i < input.length; i++){
				blocksCombinedSig[i] = new DenseNeuronArray(outputDim);
				blocksCombinedSig[i].setName("combined sig " + i);
				inference.addNeurons(combinedLevel+2, blocksCombinedSig[i]);
				inference.addMapping(new OutputMappingDenseToDense(blocksCombined[i], blocksCombinedSig[i], LogisticSigmoidLayer.singleton));				
			}
			map.setForwardParam(COMBINEDKEY, blocksCombinedSig);
		}
		else if(outputSigmoid == 2){
			DenseNeuronArray[] blocksCombinedTan = new DenseNeuronArray[input.length];		
			for(int i = 0; i < input.length; i++){
				blocksCombinedTan[i] = new DenseNeuronArray(outputDim);
				blocksCombinedTan[i].setName("combined tan " + i);
				inference.addNeurons(combinedLevel+2, blocksCombinedTan[i]);
				inference.addMapping(new OutputMappingDenseToDense(blocksCombined[i], blocksCombinedTan[i], TanSigmoidLayer.singleton));				
			}
			map.setForwardParam(COMBINEDKEY, blocksCombinedTan);
		}
		else{
			throw new RuntimeException("unknown sigmoid function");
		}
	}
	
	public void buildCombinerFinalState(DenseNeuronArray[] input, int inputStart, int inputEnd, Mapping map){
		TreeInference inference = map.getSubInference();
		RNNBlock[] blocksForward = (RNNBlock[])map.getForwardParam(FORWARDKEY);
		RNNBlock[] blocksBackward = (RNNBlock[])map.getForwardParam(BACKWARDKEY);

		int combinedLevel = inference.getMaxLevel()+1;
		DenseNeuronArray blockCombinedInput = new DenseNeuronArray(stateDim*2);
		blockCombinedInput.setName("combined final input");
		inference.addNeurons(combinedLevel+1, blockCombinedInput);
		DenseNeuronArray blockCombined = new DenseNeuronArray(outputDim);
		blockCombined.setName("combined final");
		inference.addNeurons(combinedLevel+2, blockCombined);
		DenseNeuronArray blockCombinedTan = new DenseNeuronArray(outputDim);		
		blockCombinedTan.setName("combined final tan");
		inference.addNeurons(combinedLevel+3, blockCombinedTan);

		if(type_forward != 0){
			inference.addMapping(new OutputMappingDenseToDense(0,stateDim-1, 0, stateDim-1, blocksForward[input.length-1].hState, blockCombinedInput, CopyLayer.singleton));
		}
		if(type_backward != 0){
			inference.addMapping(new OutputMappingDenseToDense(0, stateDim-1, stateDim, stateDim*2-1, blocksBackward[0].hState, blockCombinedInput, CopyLayer.singleton));
		}
		inference.addMapping(new OutputMappingDenseToDense(blockCombinedInput, blockCombined, combiner));
		inference.addMapping(new OutputMappingDenseToDense(blockCombined, blockCombinedTan, TanSigmoidLayer.singleton));

		map.setForwardParam(COMBINEDFINALBLOCKKEY, blockCombinedTan);
	}

	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray[] output, int outputStart, int outputEnd, OutputMappingDenseArrayToDenseArray map) {
		TreeInference inference = map.getSubInference();
		buildInference(input, inputStart, inputEnd, map);
		buildCombinerSequence(input, inputStart, inputEnd, map);
		inference.init();
		inference.forward();

		DenseNeuronArray[] blocks = (DenseNeuronArray[])map.getForwardParam(COMBINEDKEY);
		for(int i = 0; i < output.length; i++){
			DenseNeuronArray outputI = output[i];
			for(int d = 0; d < outputDim; d++){				
				outputI.addNeuron(d+outputStart, blocks[i].getNeuron(d));
			}
		}
	}
	
	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray output, int outputStart, int outputEnd,
			OutputMappingDenseArrayToDense map) {
		TreeInference inference = map.getSubInference();
		buildInference(input, inputStart, inputEnd, map);
		buildCombinerFinalState(input, inputStart, inputEnd, map);
		inference.init();
		inference.forward();
		DenseNeuronArray blockCombinedTan = (DenseNeuronArray)map.getForwardParam(COMBINEDFINALBLOCKKEY);		

		for(int d = 0; d < outputDim; d++){				
			output.addNeuron(d+outputStart, blockCombinedTan.getNeuron(d));
		}
	}

	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, DenseNeuronArray[] output, int outputStart,
			int outputEnd, OutputMappingDenseArrayToDenseArray map) {

		DenseNeuronArray[] blocks = (DenseNeuronArray[])map.getForwardParam(COMBINEDKEY);

		TreeInference inference = map.getSubInference();
		for(int i = 0; i < output.length; i++){
			DenseNeuronArray outputI = output[i];
			for(int d = 0; d < outputDim; d++){	
				blocks[i].addError(d, outputI.getError(d+outputStart));
			}
		}

		inference.backward();
	}	
	
	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, DenseNeuronArray output, int outputStart,
			int outputEnd, OutputMappingDenseArrayToDense mapping) {
		
		TreeInference inference = mapping.getSubInference();
		DenseNeuronArray blockCombinedTan = (DenseNeuronArray)mapping.getForwardParam(COMBINEDFINALBLOCKKEY);		

		for(int d = 0; d < outputDim; d++){	
			blockCombinedTan.addError(d, output.getError(d+outputStart));
		}

		inference.backward();				
	}

	@Override
	public void updateWeights(double learningRate, double momentum) {
		parameters.update(learningRate, momentum);
		parametersBackward.update(learningRate, momentum);
		combiner.updateWeights(learningRate, momentum);
	}	

	//	public void save(PrintStream out) {
	//		transformLayer.save(out);
	//		initialLayer.save(out);
	//	}
	//	
	//	public void load(Scanner reader) {
	//		transformLayer.load(reader);
	//		initialLayer.load(reader);
	//	}

	@Override
	public String toString() {
		String ret = "This is a " + inputDim + "x" + stateDim + "x" + outputDim + " lstm\n";
		ret+="\n combiner params:";
		ret+=combiner;
		return ret;
	}
	
	public static void main(String[] args){
		int stateDim = 200;
		int inputDim = 50;
		int instances = 1000;
		double learningRate = 0.1;
		FeatureVector.useAdagradDefault = true;
		FeatureVector.commitMethodDefault = 0;
		RNN recNN = new RNN(inputDim, stateDim, stateDim);

		int len = 5;
		double[][][] input = new double[instances][][];
		double[][][] expected = new double[instances][][];
		for(int i = 0; i < input.length; i++){
			input[i] = new double[len][];
			expected[i] = new double[len][];
			for(int j = 0; j < len; j++){
				input[i][j] = new double[inputDim];
				RandomUtils.initializeRandomArray(input[i][j], 0, 1);
				expected[i][j] = new double[stateDim];
				RandomUtils.initializeRandomArray(expected[i][j], 0, 1);				
			}
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
				states[j].setName("state " + j);
				inference.addNeurons(0, inputNeurons[j]);
				inference.addNeurons(1, states[j]);
			}
			inference.addMapping(new OutputMappingDenseArrayToDenseArray(inputNeurons, states, recNN));
			inference.init();
			for(int j = 0; j < len; j++){
				inputNeurons[j].init();
				inputNeurons[j].loadFromArray(input[i][j]);
			}
			inference.forward();
			double error = 0;
			for(int j = 0; j < len; j++){				
				states[j].computeErrorTan(expected[i][j]);
				error += states[j].sqError();
			}
			error /= len;
			inference.backward();			
			//			System.err.println("initial state");
			//			recNN.initialLayer.printWeights();
			//			System.err.println("transformation layer");
			//			recNN.transformLayer.printWeights();
			//			
			inference.commit(learningRate);
//			inference.printNeurons();
			//			recNN.printNeurons();

			PrintUtils.printDoubleArray("output = ", states[len-1].copyAsArray(), false);
			PrintUtils.printDoubleArray("error = ", states[len-1].copyErrorAsArray(), false);
			System.err.println("error " + error);
			
			double avgTime = (System.currentTimeMillis() - startTime)/(iteration+1);
			System.err.println("avg time " + avgTime);
			System.err.println("this iteration " + (System.currentTimeMillis() - iterationStart));
		}
	}
}
