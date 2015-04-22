package jnn.functions.composite.lstm.aux;

import jnn.neuron.CompositeNeuronArray;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;

public class LSTMOutputNeurons extends CompositeNeuronArray{
	public DenseNeuronArray[] states;
	public DenseNeuronArray[] cells;
	
	NeuronArray[] neurons;

	public LSTMOutputNeurons(int numStates, int stateSize) {
		this.states = new DenseNeuronArray[numStates];
		this.cells = new DenseNeuronArray[numStates];
		neurons = new NeuronArray[states.length*2];
		for(int i = 0; i < states.length; i++){
			states[i] = new DenseNeuronArray(stateSize);
			cells[i] = new DenseNeuronArray(stateSize);
			neurons[i] = states[i];
			neurons[i+states.length] = cells[i];
		}
	}
	
	public LSTMOutputNeurons(DenseNeuronArray[] states, DenseNeuronArray[] cells) {
		super();
		this.states = states;
		this.cells = cells;
		neurons = new NeuronArray[states.length*2];
		for(int i = 0; i < states.length; i++){
			neurons[i] = states[i];
			neurons[i+states.length] = cells[i];
		}
	}
	
	@Override
	public NeuronArray[] getAtomicNeurons() {
		return neurons;
	}
	
	public DenseNeuronArray getFinalState(){
		return states[states.length-1];
	}
	
	public DenseNeuronArray getFinalCell(){
		return cells[cells.length-1];
	}
}
