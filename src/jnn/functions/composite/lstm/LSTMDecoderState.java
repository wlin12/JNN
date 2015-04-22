package jnn.functions.composite.lstm;

import jnn.decoder.state.DenseNeuronState;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;

public class LSTMDecoderState extends DenseNeuronState{

	public DenseNeuronArray input;
	
	public DenseNeuronArray lstmState;
	public DenseNeuronArray lstmCell;
	
	public LSTMDecoderState(double score, boolean isFinal,
			DenseNeuronArray output, DenseNeuronArray lstmState,
			DenseNeuronArray lstmCell, DenseNeuronArray input) {
		super(score, isFinal, output);
		this.lstmState = lstmState;
		this.lstmCell = lstmCell;
		this.input = input;
	}	
	
	public static DenseNeuronArray[] getInputs(LSTMDecoderState[] states){
		DenseNeuronArray[] inputs = new DenseNeuronArray[states.length];
		for(int i = 0; i < inputs.length; i++){
			inputs[i] = states[i].input;
		}
		return inputs;
	}
	
	public static DenseNeuronArray[] getStates(LSTMDecoderState[] states){
		DenseNeuronArray[] inputs = new DenseNeuronArray[states.length];
		for(int i = 0; i < inputs.length; i++){
			inputs[i] = states[i].lstmState;
		}
		return inputs;
	}
	
	public static DenseNeuronArray[] getCells(LSTMDecoderState[] states){
		DenseNeuronArray[] inputs = new DenseNeuronArray[states.length];
		for(int i = 0; i < inputs.length; i++){
			inputs[i] = states[i].lstmCell;
		}
		return inputs;
	}
	
	public static DenseNeuronArray[] getOutputs(LSTMDecoderState[] states){
		DenseNeuronArray[] inputs = new DenseNeuronArray[states.length*2];
		for(int i = 0; i < states.length; i++){
			inputs[i] = states[i].lstmState;
			inputs[i+states.length] = states[i].lstmCell;
		}
		return inputs;
	}
	
	public static LSTMDecoderState[] buildStateSequence(DenseNeuronArray[] inputs, int stateSize){
		LSTMDecoderState[] states = new LSTMDecoderState[inputs.length];
		for(int i = 0; i < states.length; i++){
			boolean isFinal = i == states.length-1;
			DenseNeuronArray lstmState = new DenseNeuronArray(stateSize);
			lstmState.setName("lstm decoder state " + i);
			DenseNeuronArray lstmCell = new DenseNeuronArray(stateSize);
			lstmCell.setName("lstm decoder cell " + i);
			LSTMDecoderState state = new LSTMDecoderState(-1, isFinal, lstmState, lstmState, lstmCell, inputs[i]);
			states[i] = state;
		}
		return states;
	}
}
