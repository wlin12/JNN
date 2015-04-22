package jnn.decoder.state;

import jnn.neuron.DenseNeuronArray;

public class DenseNeuronState extends DecoderState{
	public DenseNeuronArray output;

	public DenseNeuronState(double score, boolean isFinal,
			DenseNeuronArray output) {
		super(score, isFinal);
		this.output = output;
	}	
}
