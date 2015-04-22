package jnn.functions.composite.lstm;

import java.util.List;

import jnn.decoder.DecoderInterface;
import jnn.decoder.stackbased.StackBasedDecoder;
import jnn.decoder.state.DecoderState;
import jnn.functions.composite.lstm.aux.LSTMBlock;
import jnn.functions.composite.lstm.aux.LSTMMapping;
import jnn.functions.composite.lstm.aux.LSTMParameters;
import jnn.functions.composite.lstm.aux.LSTMStateTransform;
import jnn.functions.nonparametrized.CopyLayer;
import jnn.functions.nonparametrized.SoftmaxLayer;
import jnn.functions.nonparametrized.TanSigmoidLayer;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.training.GraphInference;

public class LSTMDecoderWithModel2Alignment extends Layer implements LSTMStateTransform{
	LSTMParameters parameters;
	
	private static final String STATES = "states";

	int inputDim;
	int stateDim;
	int sourceDim;
	double ratio;

	public LSTMDecoderWithModel2Alignment(int inputDim, int sourceDim, int stateDim, int inputAvgLength, int sourceAvgLength) {

		this.inputDim = inputDim;
		this.stateDim = stateDim;
		this.sourceDim = sourceDim;
		this.ratio = sourceAvgLength/(double)inputAvgLength;
		parameters = new LSTMParameters(inputDim+sourceDim, stateDim);
	}

	public LSTMBlock[] getForwardBlocks(int start, DenseNeuronArray[] input, DenseNeuronArray[] source, DenseNeuronArray initialState, DenseNeuronArray initialCell, GraphInference inference){
		// add input units
		inference.addNeurons(0, input);
		inference.addNeurons(0, initialState);
		inference.addNeurons(0, initialCell);

		int startLevel = 3;
		
		LSTMBlock currentBlock = new LSTMBlock(initialState, initialCell, startLevel);
		LSTMBlock[] blocks = new LSTMBlock[input.length];

		for(int i = 0; i < input.length; i++){
			buildForwardBlock(start+i, input[i], source, currentBlock, inference);
			blocks[i] = currentBlock;
			currentBlock = currentBlock.nextState();
		}
		
		return blocks;
	}
	
	public void buildForwardBlock(int pos, DenseNeuronArray input, DenseNeuronArray[] source, LSTMBlock block, GraphInference inference){
		int start = block.start;
		int sourcePos = (int)Math.round(ratio*pos);
		if(source!=null && sourcePos>=source.length){
			sourcePos = source.length-1;
		}
		DenseNeuronArray inputWithSource = new DenseNeuronArray(inputDim+sourceDim);
		inputWithSource.setName("input with source " + pos);
		inference.addNeurons(start,inputWithSource);
		if(source!=null){
			inference.addMapping(new OutputMappingDenseToDense(0,sourceDim-1,0,sourceDim-1,source[sourcePos], inputWithSource, CopyLayer.singleton));
		}
		inference.addMapping(new OutputMappingDenseToDense(0,inputDim-1,sourceDim, sourceDim+inputDim-1,input, inputWithSource, CopyLayer.singleton));
		start++;
		block.start = start;
		block.addToInference(inference, inputWithSource, parameters);
	}
	
	public void buildInference(DenseNeuronArray[] input, DenseNeuronArray[] source, DenseNeuronArray initialState, DenseNeuronArray initialCell, Mapping map){
		GraphInference inference = map.getSubInference();
		if(source != null){
			inference.addNeurons(0, source);
		}
		LSTMBlock[] blocksForward = getForwardBlocks(0,input, source, initialState, initialCell, inference);
		map.setForwardParam(STATES, blocksForward);
	}

	@Override
	public void forward(LSTMMapping map) {
		GraphInference inference = map.getSubInference();
		
		DenseNeuronArray[] x = LSTMDecoderState.getInputs(map.states);
		buildInference(x,map.sources, map.initialState, map.initialCell, map);
		inference.init();
		inference.forward();
		
		DenseNeuronArray[] lstmStates = LSTMDecoderState.getStates(map.states);
		DenseNeuronArray[] lstmCells = LSTMDecoderState.getCells(map.states);

		LSTMBlock[] blocks = (LSTMBlock[])map.getForwardParam(STATES);
		for(int i = 0; i < lstmStates.length; i++){
			DenseNeuronArray outputI = lstmStates[i];
			DenseNeuronArray cellI = lstmCells[i];
			for(int d = 0; d < stateDim; d++){				
				outputI.addNeuron(d, blocks[i].hState.getNeuron(d));
				cellI.addNeuron(d, blocks[i].cMen.getNeuron(d));
			}
		}
	}

	@Override
	public void backward(LSTMMapping map) {
		LSTMBlock[] blocks = (LSTMBlock[])map.getForwardParam(STATES);

		DenseNeuronArray[] lstmStates = LSTMDecoderState.getStates(map.states);
		DenseNeuronArray[] lstmCells = LSTMDecoderState.getCells(map.states);

		GraphInference inference = map.getSubInference();
		for(int i = 0; i < lstmStates.length; i++){
			DenseNeuronArray outputI = lstmStates[i];
			DenseNeuronArray cellI = lstmCells[i];
			for(int d = 0; d < stateDim; d++){	
				blocks[i].hState.addError(d, outputI.getError(d));
				blocks[i].cMen.addError(d, cellI.getError(d));
			}
		}
		inference.backward();
	}
	
	public LSTMDecoderWithAlignmentState decode(LSTMDecoderWithAlignmentState initialDecoderState, DenseNeuronArray[] source, int beam, DecoderInterface scorer){
		
		StackBasedDecoder decoder = new StackBasedDecoder(new DecoderInterface() {
			
			@Override
			public List<DecoderState> expand(DecoderState state) {
				LSTMDecoderWithAlignmentState lstmState = (LSTMDecoderWithAlignmentState) state;
				GraphInference inference = new GraphInference(0, false);
				inference.addNeurons(0,source);
				LSTMBlock block = getForwardBlocks(state.numberOfPrevStates, new DenseNeuronArray[]{lstmState.input}, source, lstmState.lstmState, lstmState.lstmCell, inference)[0];
				inference.init();
				inference.forward();
				lstmState.output = block.hState;
				
				List<DecoderState> states = scorer.expand(state);
				for(DecoderState nextState : states){
					LSTMDecoderWithAlignmentState nextLstmState = (LSTMDecoderWithAlignmentState) nextState;
					nextLstmState.lstmState = block.hState;
					nextLstmState.lstmCell = block.cMen;
				}
				return states;
			}
		}, initialDecoderState);
		decoder.stackSize = beam;
		decoder.decode();
		return (LSTMDecoderWithAlignmentState)decoder.getBestState();
	}
			
	@Override
	public void updateWeights(double learningRate, double momentum) {
		parameters.update(learningRate, momentum);
	}
	
	public int getStateDim() {
		return stateDim;
	}
}
