package jnn.functions.composite;

import jnn.features.FeatureVector;
import jnn.functions.DenseArrayToDenseArrayTransform;
import jnn.functions.DenseArrayToDenseTransform;
import jnn.functions.nonparametrized.LogisticSigmoidLayer;
import jnn.functions.nonparametrized.TanSigmoidLayer;
import jnn.functions.parametrized.CopyLayer;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.HadamardProductLayer;
import jnn.functions.parametrized.StaticLayer;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.training.TreeInference;
import util.PrintUtils;
import util.RandomUtils;

public class LongShortTermMemoryRNNVariableState extends RNN implements DenseArrayToDenseArrayTransform, DenseArrayToDenseTransform{

	private class LSTMParameters {
		DenseFullyConnectedLayer inputTransformLayer;
		DenseFullyConnectedLayer forgetTransformLayer;
		DenseFullyConnectedLayer outputTransformLayer;
		DenseFullyConnectedLayer cellTransformLayer;
		StaticLayer initialStateLayer;
		StaticLayer initialCellLayer;
		
		public LSTMParameters(int inputDim, int stateDim) {
			inputTransformLayer = new DenseFullyConnectedLayer(inputDim+stateDim, stateDim);
			inputTransformLayer.initialize(true, false);
			forgetTransformLayer = new DenseFullyConnectedLayer(inputDim+stateDim, stateDim);
			forgetTransformLayer.initialize(true, false);
			outputTransformLayer = new DenseFullyConnectedLayer(inputDim+stateDim, stateDim);
			outputTransformLayer.initialize(true, false);
			cellTransformLayer = new DenseFullyConnectedLayer(inputDim+stateDim, stateDim);
			cellTransformLayer.initialize(false, true);
			initialStateLayer = new StaticLayer(stateDim);
			initialCellLayer = new StaticLayer(stateDim);
		}
		
		public void update(double learningRate, double momentum){
			inputTransformLayer.updateWeights(learningRate, momentum);
			forgetTransformLayer.updateWeights(learningRate, momentum);
			outputTransformLayer.updateWeights(learningRate, momentum);
			cellTransformLayer.updateWeights(learningRate, momentum);
			initialCellLayer.updateWeights(learningRate, momentum);
			initialStateLayer.updateWeights(learningRate, momentum);

		}
	}
	
	private class LSTMBlock {
		DenseNeuronArray hprevState;

		DenseNeuronArray iGate;
		DenseNeuronArray fGate;
		DenseNeuronArray oGate;
		DenseNeuronArray cGate;

		DenseNeuronArray cprevMem;
		DenseNeuronArray cMen;

		DenseNeuronArray hState;

		int start;
		int end;

		public LSTMBlock(DenseNeuronArray hprevState,
				DenseNeuronArray cprevMem, int start) {
			super();
			this.hprevState = hprevState;
			this.cprevMem = cprevMem;
			this.start = start;
		}

		public LSTMBlock nextState(){
			return new LSTMBlock(hState, cMen, end+1);
		}

		public void addToInference(TreeInference inference, DenseNeuronArray inputX, LSTMParameters parameters){
			int units = inputX.len();
			int stateSize = hprevState.size;

			int level = start;
			
			// build input output and forget input data
			DenseNeuronArray iGateInput = new DenseNeuronArray(units + stateSize);
			iGateInput.setName("input gate input: level " + start);
			inference.addNeurons(level, iGateInput);
			
			inference.addMapping(new OutputMappingDenseToDense(0,units-1,0,units-1,inputX, iGateInput, CopyLayer.singleton));
			inference.addMapping(new OutputMappingDenseToDense(0,stateSize-1,units,units+stateSize-1,hprevState, iGateInput, CopyLayer.singleton));

			DenseNeuronArray fGateInput = new DenseNeuronArray(units + stateSize);
			fGateInput.setName("forget gate input: level " + start);
			inference.addNeurons(level, fGateInput);
			
			inference.addMapping(new OutputMappingDenseToDense(0,units-1,0,units-1,inputX, fGateInput, CopyLayer.singleton));
			inference.addMapping(new OutputMappingDenseToDense(0,stateSize-1,units,units+stateSize-1,hprevState, fGateInput, CopyLayer.singleton));

			DenseNeuronArray oGateInput = new DenseNeuronArray(units + stateSize);
			oGateInput.setName("output gate input: level " + start);
			inference.addNeurons(level, oGateInput);
			
			inference.addMapping(new OutputMappingDenseToDense(0,units-1,0,units-1,inputX, oGateInput, CopyLayer.singleton));
			inference.addMapping(new OutputMappingDenseToDense(0,stateSize-1,units,units+stateSize-1,hprevState, oGateInput, CopyLayer.singleton));

			DenseNeuronArray cGateInput = new DenseNeuronArray(units + stateSize);
			cGateInput.setName("cell gate input: level " + start);
			inference.addNeurons(level, cGateInput);
			
			inference.addMapping(new OutputMappingDenseToDense(0,units-1,0,units-1,inputX, cGateInput, CopyLayer.singleton));
			inference.addMapping(new OutputMappingDenseToDense(0,stateSize-1,units,units+stateSize-1,hprevState, cGateInput, CopyLayer.singleton));			
			
			// build input output and forget gates
			level++;
			iGate = new DenseNeuronArray(stateSize);
			iGate.setName("input gate: level " + start);
			inference.addNeurons(level, iGate);

			inference.addMapping(new OutputMappingDenseToDense(iGateInput, iGate, parameters.inputTransformLayer));

			fGate = new DenseNeuronArray(stateSize);
			fGate.setName("forget gate: level " + start);
			inference.addNeurons(level, fGate);

			inference.addMapping(new OutputMappingDenseToDense(fGateInput, fGate, parameters.forgetTransformLayer));			

			oGate = new DenseNeuronArray(stateSize);
			oGate.setName("output gate: level " + start);
			inference.addNeurons(level, oGate);

			inference.addMapping(new OutputMappingDenseToDense(oGateInput, oGate, parameters.outputTransformLayer));

			cGate = new DenseNeuronArray(stateSize);
			cGate.setName("cell gate: level " + start);
			inference.addNeurons(level, cGate);

			inference.addMapping(new OutputMappingDenseToDense(cGateInput, cGate, parameters.cellTransformLayer));

			//apply sigmoid function
			level++;
			DenseNeuronArray iGateSig = new DenseNeuronArray(stateSize);
			iGateSig.setName("input gate sig: level " + start);
			inference.addNeurons(level, iGateSig);
			inference.addMapping(new OutputMappingDenseToDense(iGate, iGateSig, LogisticSigmoidLayer.singleton));

			DenseNeuronArray fGateSig = new DenseNeuronArray(stateSize);
			fGateSig.setName("forget gate sig: level " + start);
			inference.addNeurons(level, fGateSig);
			inference.addMapping(new OutputMappingDenseToDense(fGate, fGateSig, LogisticSigmoidLayer.singleton));

			DenseNeuronArray oGateSig = new DenseNeuronArray(stateSize);
			oGateSig.setName("output gate sig: level " + start);
			inference.addNeurons(level, oGateSig);			
			inference.addMapping(new OutputMappingDenseToDense(oGate, oGateSig, LogisticSigmoidLayer.singleton));

			DenseNeuronArray cGateTanh = new DenseNeuronArray(stateSize);
			cGateTanh.setName("cell gate tanh: level " + start);
			inference.addNeurons(level, cGateTanh);			
			inference.addMapping(new OutputMappingDenseToDense(cGate, cGateTanh, TanSigmoidLayer.singleton));

			//next cell computation
			level++;
			cMen = new DenseNeuronArray(stateSize);
			cMen.setName("cell memory: level " + start);
			inference.addNeurons(level, cMen);

			inference.addMapping(new OutputMappingDenseArrayToDense(new DenseNeuronArray[]{iGateSig,cGateTanh} , cMen, HadamardProductLayer.singleton));			
			inference.addMapping(new OutputMappingDenseArrayToDense(new DenseNeuronArray[]{fGateSig,cprevMem} , cMen, HadamardProductLayer.singleton));			

			//apply sigmoid
			level++;
			DenseNeuronArray cMenTanh = new DenseNeuronArray(stateSize);
			cMenTanh.setName("cell memory sig: level " + start);
			inference.addNeurons(level, cMenTanh);
			inference.addMapping(new OutputMappingDenseToDense(cMen, cMenTanh, TanSigmoidLayer.singleton));

			//next state computation
			level++;
			hState = new DenseNeuronArray(stateSize);
			hState.setName("state at level " + start);
			inference.addNeurons(level, hState);
			inference.addMapping(new OutputMappingDenseArrayToDense(new DenseNeuronArray[]{oGateSig,cMenTanh} , hState, HadamardProductLayer.singleton));

			end = level+1;
		}
	}

	LSTMParameters parameters;
	LSTMParameters parametersBackward;

	//combiner Layer
	DenseFullyConnectedLayer combiner;
	
	private static final String FORWARDKEY = "forwardblocks";
	private static final String BACKWARDKEY = "backwardblocks";
	private static final String COMBINEDKEY = "combinedblocks";
	private static final String COMBINEDFINALBLOCKKEY = "combinedfinalblock";

	public LongShortTermMemoryRNNVariableState(int inputDim, int stateDim, int outputDim) {
		super(inputDim, stateDim, outputDim);
		
		parameters = new LSTMParameters(inputDim, stateDim);
		parametersBackward = new LSTMParameters(inputDim, stateDim);

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
			
			DenseNeuronArray initialCell = new DenseNeuronArray(stateDim);
			inference.addNeurons(1, initialCell);

//			OutputMappingVoidToDense initialStateMapping = new OutputMappingVoidToDense(initialState, parameters.initialStateLayer);
//			inference.addMapping(0, 1, initialStateMapping);
//
//			OutputMappingVoidToDense initialCellMapping = new OutputMappingVoidToDense(initialCell, parameters.initialCellLayer);
//			inference.addMapping(0, 1, initialCellMapping);

			LSTMBlock blockForward = new LSTMBlock(initialState, initialCell, startLevel);
			LSTMBlock[] blocksForward = new LSTMBlock[input.length];
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

			DenseNeuronArray initialCell = new DenseNeuronArray(stateDim);
			inference.addNeurons(1, initialCell);

//			OutputMappingVoidToDense initialStateMapping = new OutputMappingVoidToDense(initialState, parametersBackward.initialStateLayer);
//			inference.addMapping(0, 1, initialStateMapping);
//
//			OutputMappingVoidToDense initialCellMapping = new OutputMappingVoidToDense(initialCell, parametersBackward.initialCellLayer);
//			inference.addMapping(0, 1, initialCellMapping);

			LSTMBlock blockBackward = new LSTMBlock(initialState, initialCell, startLevel);
			LSTMBlock[] blocksBackward = new LSTMBlock[input.length];
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
		LSTMBlock[] blocksForward = (LSTMBlock[])map.getForwardParam(FORWARDKEY);
		LSTMBlock[] blocksBackward = (LSTMBlock[])map.getForwardParam(BACKWARDKEY);

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
		LSTMBlock[] blocksForward = (LSTMBlock[])map.getForwardParam(FORWARDKEY);
		LSTMBlock[] blocksBackward = (LSTMBlock[])map.getForwardParam(BACKWARDKEY);

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
		LongShortTermMemoryRNNVariableState recNN = new LongShortTermMemoryRNNVariableState(inputDim, stateDim, stateDim);

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
