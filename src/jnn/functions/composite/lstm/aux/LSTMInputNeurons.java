package jnn.functions.composite.lstm.aux;

import jnn.neuron.CompositeNeuronArray;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;

public class LSTMInputNeurons extends CompositeNeuronArray{
	public DenseNeuronArray[] inputs;
	public DenseNeuronArray initialState;
	public DenseNeuronArray initialCell;
	public DenseNeuronArray[] source;

	public LSTMInputNeurons(DenseNeuronArray[] inputs,
			DenseNeuronArray initialState, DenseNeuronArray initialCell) {
		super();
		this.inputs = inputs;
		this.initialState = initialState;
		this.initialCell = initialCell;		
	}

	public LSTMInputNeurons(DenseNeuronArray[] inputs, int stateSize) {
		super();
		this.inputs = inputs;
		this.initialState = new DenseNeuronArray(stateSize);
		this.initialCell = new DenseNeuronArray(stateSize);		
	}

	@Override
	public NeuronArray[] getAtomicNeurons() {
		int numberOfNeuronUnits = inputs.length + 2;
		if(source!=null){
			numberOfNeuronUnits += source.length;
		}
		NeuronArray[] neurons = new NeuronArray[numberOfNeuronUnits];
		neurons[0] = initialState;
		neurons[1] = initialCell;
		for(int i = 0; i < inputs.length; i++){
			neurons[2+i] = inputs[i];
		}
		if(source!=null){
			for(int i = 0; i < source.length; i++){
				neurons[2+inputs.length+i] = source[i];
			}
		}
		return neurons;
	}

	public DenseNeuronArray getInitialCell() {
		return initialCell;
	}

	public DenseNeuronArray getInitialState() {
		return initialState;
	}

	public void setSource(DenseNeuronArray[] source) {
		this.source = source;
	}
}
