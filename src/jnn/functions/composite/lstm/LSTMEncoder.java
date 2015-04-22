package jnn.functions.composite.lstm;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;

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

public class LSTMEncoder extends Layer implements LSTMStateTransform{
	LSTMParameters parameters;

	private static final String FINALBLOCK = "finalblock";

	int inputDim;
	int stateDim;

	public LSTMEncoder(int inputDim, int stateDim) {

		this.inputDim = inputDim;
		this.stateDim = stateDim;
		parameters = new LSTMParameters(inputDim, stateDim);		
	}

	public void buildInference(DenseNeuronArray[] input, DenseNeuronArray initialState, DenseNeuronArray initialCell, Mapping map){
		GraphInference inference = map.getSubInference();

		// add input units
		for(int i = 0; i < input.length; i++){
			inference.addNeurons(0, input[i]);
		}
		int startLevel = 2;

		inference.addNeurons(0, initialState);
		inference.addNeurons(0, initialCell);

		LSTMBlock blockForward = new LSTMBlock(initialState, initialCell, startLevel);
		LSTMBlock[] blocksForward = blockForward.addMultipleBlocks(inference, input, parameters);
		map.setForwardParam(FINALBLOCK, blocksForward[blocksForward.length-1]);

	}

	@Override
	public void forward(LSTMMapping map) {
		GraphInference inference = map.getSubInference();
		DenseNeuronArray[] x = LSTMDecoderState.getInputs(map.states);
		buildInference(x, map.initialState, map.initialCell, map);
		inference.init();
		inference.forward();

		LSTMBlock finalBlock = (LSTMBlock)map.getForwardParam(FINALBLOCK);

		DenseNeuronArray outputI = map.states[map.states.length-1].lstmState;
		DenseNeuronArray cellI = map.states[map.states.length-1].lstmCell;
		for(int d = 0; d < stateDim; d++){				
			outputI.addNeuron(d, finalBlock.hState.getNeuron(d));
			cellI.addNeuron(d, finalBlock.cMen.getNeuron(d));
		}
	}

	@Override
	public void backward(LSTMMapping map) {
		LSTMBlock finalBlock = (LSTMBlock)map.getForwardParam(FINALBLOCK);

		GraphInference inference = map.getSubInference();
		DenseNeuronArray outputI = map.states[map.states.length-1].lstmState;
		DenseNeuronArray cellI = map.states[map.states.length-1].lstmCell;
		for(int d = 0; d < stateDim; d++){	
			finalBlock.hState.addError(d, outputI.getError(d));
			finalBlock.cMen.addError(d, cellI.getError(d));
		}
		inference.backward();
	}
	
	@Override
	public void updateWeights(double learningRate, double momentum) {
		parameters.update(learningRate, momentum);
	}
	
	public int getStateDim() {
		return stateDim;
	}
	
	public void save(PrintStream out){
		out.println("lstm encoder parameters");
		out.println(inputDim);
		out.println(stateDim);
		parameters.save(out);
	}

	public static LSTMEncoder load(BufferedReader in){
		try{			
			in.readLine();
			int inputDim = Integer.parseInt(in.readLine());
			int stateDim = Integer.parseInt(in.readLine());
			LSTMEncoder encoder = new LSTMEncoder(inputDim, stateDim);
			encoder.parameters = LSTMParameters.load(in);
			return encoder;
		}
		catch (IOException e){
			throw new RuntimeException(e);
		}
	}
}
