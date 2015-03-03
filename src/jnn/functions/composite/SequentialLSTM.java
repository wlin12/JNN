package jnn.functions.composite;

import jnn.functions.DenseArrayToDenseArrayTransform;
import jnn.functions.nonparametrized.LogisticSigmoidLayer;
import jnn.functions.nonparametrized.TanSigmoidLayer;
import jnn.functions.parametrized.CopyLayer;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.HadamardProductLayer;
import jnn.functions.parametrized.Layer;
import jnn.functions.parametrized.StaticLayer;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.training.TreeInference;

public class SequentialLSTM extends Layer implements DenseArrayToDenseArrayTransform{

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
		
		public void print(){
			System.err.println(inputTransformLayer);
			System.err.println(forgetTransformLayer);
			System.err.println(outputTransformLayer);
			System.err.println(cellTransformLayer);
			System.err.println(initialCellLayer);
			System.err.println(initialStateLayer);
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

	LSTMParameters parametersEncoder;
	LSTMParameters parametersEncoderBackward;
	LSTMParameters parametersDecoder;
	LSTMParameters parametersDecoderBackward;

	//combiner Layer
	DenseFullyConnectedLayer combiner;
	
	int inputDim;
	int stateDim;
	int outputDim;

	public int type_forward = 1; // 0 -> do not use forward
	public int type_backward = 1; // 0 -> do not use backward
	
	public int outputSigmoid = 2; // 0 -> none, 1 -> sigmoid, 2 -> tanh;
	private static final String FORWARDKEY = "forwardblocks";
	private static final String BACKWARDKEY = "backwardblocks";
	private static final String COMBINEDKEY = "combinedblocks";

	public SequentialLSTM(int inputDim, int stateDim, int outputDim) {
		this.inputDim = inputDim;
		this.stateDim = stateDim;
		this.outputDim = outputDim;
		
		parametersEncoder = new LSTMParameters(inputDim, stateDim);
		parametersEncoderBackward = new LSTMParameters(inputDim, stateDim);
		parametersDecoder = new LSTMParameters(inputDim, stateDim);
		parametersDecoderBackward = new LSTMParameters(inputDim, stateDim);

		combiner = new DenseFullyConnectedLayer(stateDim*2, outputDim);
		combiner.initializeForTanhSigmoid();

	}
	
	public void buildInference(DenseNeuronArray[] input, int decodeStart, int inputStart, int inputEnd, Mapping map){
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

//			OutputMappingVoidToDense initialStateMapping = new OutputMappingVoidToDense(initialState, parametersEncoder.initialStateLayer);
//			inference.addMapping(0, 1, initialStateMapping);
//
//			OutputMappingVoidToDense initialCellMapping = new OutputMappingVoidToDense(initialCell, parametersEncoder.initialCellLayer);
//			inference.addMapping(0, 1, initialCellMapping);

			LSTMBlock blockForward = new LSTMBlock(initialState, initialCell, startLevel);
			LSTMBlock[] blocksForward = new LSTMBlock[input.length];
			for(int i = 0; i < decodeStart; i++){
				blockForward.addToInference(inference, input[i], parametersEncoder);
				blocksForward[i] = blockForward;
				blockForward = blockForward.nextState();
			}
			for(int i = decodeStart; i<input.length; i++){
				blockForward.addToInference(inference, input[i], parametersDecoder);
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

//			OutputMappingVoidToDense initialStateMapping = new OutputMappingVoidToDense(initialState, parametersEncoderBackward.initialStateLayer);
//			inference.addMapping(0, 1, initialStateMapping);
//
//			OutputMappingVoidToDense initialCellMapping = new OutputMappingVoidToDense(initialCell, parametersEncoderBackward.initialCellLayer);
//			inference.addMapping(0, 1, initialCellMapping);

			LSTMBlock blockBackward = new LSTMBlock(initialState, initialCell, startLevel);
			LSTMBlock[] blocksBackward = new LSTMBlock[input.length];
			for(int i = 0; i < decodeStart; i++){
				blockBackward.addToInference(inference, input[decodeStart-i-1], parametersEncoderBackward);
				blocksBackward[decodeStart-i-1] = blockBackward;
				blockBackward = blockBackward.nextState();
			}
			for(int i = decodeStart; i<input.length; i++){
				blockBackward.addToInference(inference, input[input.length-i-1], parametersDecoderBackward);
				blocksBackward[input.length-i-1+decodeStart] = blockBackward;
				blockBackward = blockBackward.nextState();				
			}
			map.setForwardParam(BACKWARDKEY, blocksBackward);
		}
	}
	
	public void buildCombinerSequence(DenseNeuronArray[] input, int decodeStart, int inputStart, int inputEnd, Mapping map){
		TreeInference inference = map.getSubInference();
		LSTMBlock[] blocksForward = (LSTMBlock[])map.getForwardParam(FORWARDKEY);
		LSTMBlock[] blocksBackward = (LSTMBlock[])map.getForwardParam(BACKWARDKEY);

		int numberOfOutputNodes = input.length - decodeStart + 1;
		DenseNeuronArray[] blocksCombinedInput = new DenseNeuronArray[numberOfOutputNodes];
		DenseNeuronArray[] blocksCombined = new DenseNeuronArray[numberOfOutputNodes];
		int combinedLevel = inference.getMaxLevel()+1;

		for(int i = decodeStart; i < input.length; i++){
			blocksCombinedInput[i-decodeStart] = new DenseNeuronArray(stateDim*2);
			blocksCombinedInput[i-decodeStart].setName("combined input " + i);
			inference.addNeurons(combinedLevel, blocksCombinedInput[i-decodeStart]);
			blocksCombined[i-decodeStart] = new DenseNeuronArray(outputDim);
			blocksCombined[i-decodeStart].setName("combined " + i);
			inference.addNeurons(combinedLevel+1, blocksCombined[i-decodeStart]);
			DenseNeuronArray forwardBlock = blocksForward[i].hprevState;
			DenseNeuronArray backwardBlock = blocksBackward[i].hprevState;

			if(type_forward != 0){
				inference.addMapping(new OutputMappingDenseToDense(0, stateDim-1, 0, stateDim-1, forwardBlock, blocksCombinedInput[i-decodeStart], CopyLayer.singleton));
			}
			if(type_backward != 0){
				inference.addMapping(new OutputMappingDenseToDense(0, stateDim-1, stateDim, stateDim*2-1, backwardBlock, blocksCombinedInput[i-decodeStart], CopyLayer.singleton));
			}
			inference.addMapping(new OutputMappingDenseToDense(blocksCombinedInput[i-decodeStart], blocksCombined[i-decodeStart], combiner));
		}
		if(outputSigmoid == 0){
			map.setForwardParam(COMBINEDKEY, blocksCombined);
		}
		else if(outputSigmoid == 1){
			DenseNeuronArray[] blocksCombinedSig = new DenseNeuronArray[numberOfOutputNodes];		
			for(int i = decodeStart; i < input.length; i++){
				blocksCombinedSig[i-decodeStart] = new DenseNeuronArray(outputDim);
				blocksCombinedSig[i-decodeStart].setName("combined sig " + i);
				inference.addNeurons(combinedLevel+2, blocksCombinedSig[i-decodeStart]);
				inference.addMapping(new OutputMappingDenseToDense(blocksCombined[i-decodeStart], blocksCombinedSig[i-decodeStart], LogisticSigmoidLayer.singleton));				
			}
			map.setForwardParam(COMBINEDKEY, blocksCombinedSig);
		}
		else if(outputSigmoid == 2){
			DenseNeuronArray[] blocksCombinedTan = new DenseNeuronArray[numberOfOutputNodes];		
			for(int i = decodeStart; i < input.length; i++){
				blocksCombinedTan[i-decodeStart] = new DenseNeuronArray(outputDim);
				blocksCombinedTan[i-decodeStart].setName("combined tan " + i);
				inference.addNeurons(combinedLevel+2, blocksCombinedTan[i-decodeStart]);
				inference.addMapping(new OutputMappingDenseToDense(blocksCombined[i-decodeStart], blocksCombinedTan[i-decodeStart], TanSigmoidLayer.singleton));				
			}
			map.setForwardParam(COMBINEDKEY, blocksCombinedTan);
		}
		else{
			throw new RuntimeException("unknown sigmoid function");
		}
	}

	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray[] output, int outputStart, int outputEnd, OutputMappingDenseArrayToDenseArray map) {
		TreeInference inference = map.getSubInference();
		int decoderStart = input.length - output.length;
		buildInference(input, decoderStart, inputStart, inputEnd, map);
		buildCombinerSequence(input, decoderStart, inputStart, inputEnd, map);
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
	public void updateWeights(double learningRate, double momentum) {
		parametersEncoder.update(learningRate, momentum);
		parametersEncoderBackward.update(learningRate, momentum);
		parametersDecoder.update(learningRate, momentum);
		parametersDecoderBackward.update(learningRate, momentum);
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
		parametersEncoder.print();
		parametersEncoderBackward.print();
		parametersDecoder.print();
		parametersDecoderBackward.print();
		ret += combiner;
		return ret;
	}
	
	public static void main(String[] args){
		
	}
}
