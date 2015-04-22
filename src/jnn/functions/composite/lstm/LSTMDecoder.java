package jnn.functions.composite.lstm;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import jnn.decoder.DecoderInterface;
import jnn.decoder.stackbased.StackBasedDecoder;
import jnn.decoder.state.DecoderState;
import jnn.functions.composite.lstm.aux.LSTMBlock;
import jnn.functions.composite.lstm.aux.LSTMInputNeurons;
import jnn.functions.composite.lstm.aux.LSTMMapping;
import jnn.functions.composite.lstm.aux.LSTMOutputNeurons;
import jnn.functions.composite.lstm.aux.LSTMParameters;
import jnn.functions.composite.lstm.aux.LSTMStateTransform;
import jnn.functions.parametrized.Layer;
import jnn.mapping.Mapping;
import jnn.neuron.DenseNeuronArray;
import jnn.training.GraphInference;

public class LSTMDecoder extends Layer implements LSTMStateTransform{
	LSTMParameters parameters;

	private static final String STATES = "states";

	int inputDim;
	int stateDim;

	public LSTMDecoder(int inputDim, int stateDim) {

		this.inputDim = inputDim;
		this.stateDim = stateDim;
		parameters = new LSTMParameters(inputDim, stateDim);		
	}

	public LSTMBlock[] getForwardBlocks(DenseNeuronArray[] input, DenseNeuronArray initialState, DenseNeuronArray initialCell, GraphInference inference){
		// add input units
		for(int i = 0; i < input.length; i++){
			inference.addNeurons(0, input[i]);
		}
		int startLevel = 2;

		inference.addNeurons(0, initialState);
		inference.addNeurons(0, initialCell);

		LSTMBlock blockForward = new LSTMBlock(initialState, initialCell, startLevel);
		LSTMBlock[] blocksForward = blockForward.addMultipleBlocks(inference, input, parameters);
		return blocksForward;
	}

	public void buildInference(DenseNeuronArray[] input, DenseNeuronArray initialState, DenseNeuronArray initialCell, Mapping map){
		GraphInference inference = map.getSubInference();
		LSTMBlock[] blocksForward = getForwardBlocks(input, initialState, initialCell, inference);
		map.setForwardParam(STATES, blocksForward);
	}

	@Override
	public void forward(LSTMMapping map) {
		GraphInference inference = map.getSubInference();

		DenseNeuronArray[] x = LSTMDecoderState.getInputs(map.states);
		buildInference(x, map.initialState, map.initialCell, map);
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

	public StackBasedDecoder runDecoder(LSTMDecoderState initialDecoderState, int beam, DecoderInterface scorer){
		StackBasedDecoder decoder = new StackBasedDecoder(new DecoderInterface() {

			@Override
			public List<DecoderState> expand(DecoderState state) {
				LSTMDecoderState lstmState = (LSTMDecoderState) state;
				GraphInference inference = new GraphInference(0, false);
				LSTMBlock block = getForwardBlocks(new DenseNeuronArray[]{lstmState.input}, lstmState.lstmState, lstmState.lstmCell, inference)[0];
				inference.init();
				inference.forward();
				lstmState.output = block.hState;

				List<DecoderState> states = scorer.expand(state);
				for(DecoderState nextState : states){
					LSTMDecoderState nextLstmState = (LSTMDecoderState) nextState;
					nextLstmState.lstmState = block.hState;
					nextLstmState.lstmCell = block.cMen;
				}
				return states;
			}
		}, initialDecoderState);
		decoder.stackSize = beam;
		decoder.decode();
		return decoder;
	}

	public LSTMDecoderState decode(LSTMDecoderState initialDecoderState, int beam, DecoderInterface scorer){

		StackBasedDecoder decoder = runDecoder(initialDecoderState, beam, scorer);
		LinkedList<DecoderState> ret = decoder.getBestStates(1);
		if(ret.size() == 0){
			return null;
		}
		else{
			return (LSTMDecoderState)ret.getFirst();
		}
	}

	public LSTMDecoderState[] decode(LSTMDecoderState initialDecoderState, int beam, int topN, DecoderInterface scorer){

		StackBasedDecoder decoder = runDecoder(initialDecoderState, beam, scorer);
		LinkedList<DecoderState> states = decoder.getBestStates(topN);
		LSTMDecoderState[] ret = new LSTMDecoderState[states.size()];
		Iterator<DecoderState> it = states.iterator();
		for(int i = 0; i < states.size(); i++){
			ret[i] = (LSTMDecoderState)it.next();
		}
		return ret;
	}

	public void save(PrintStream out){
		out.println(inputDim);
		out.println(stateDim);
		parameters.save(out);
	}

	public static LSTMDecoder load(BufferedReader in){
		try{
			int inputDim = Integer.parseInt(in.readLine());
			int stateDim = Integer.parseInt(in.readLine());
			LSTMDecoder decoder = new LSTMDecoder(inputDim, stateDim);
			decoder.parameters = LSTMParameters.load(in);
			return decoder;
		}
		catch (IOException e){
			throw new RuntimeException(e);
		}
	}

	@Override
	public void updateWeights(double learningRate, double momentum) {
		parameters.update(learningRate, momentum);		
	}

	public int getStateDim() {
		return stateDim;
	}
}
