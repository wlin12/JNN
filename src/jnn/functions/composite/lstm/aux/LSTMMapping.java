package jnn.functions.composite.lstm.aux;

import jnn.functions.composite.lstm.LSTMDecoderState;
import jnn.functions.parametrized.Layer;
import jnn.mapping.Mapping;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;

public class LSTMMapping extends Mapping{

	public LSTMDecoderState[] states;
	public LSTMStateTransform layer;	
	public DenseNeuronArray initialState;
	public DenseNeuronArray initialCell;
	
	public DenseNeuronArray[] sources;
	
	private DenseNeuronArray[] inputs;
	private DenseNeuronArray[] outputs;
	
	
	public LSTMMapping(LSTMDecoderState[] states,
			LSTMStateTransform layer) {
		super(0, 1, 0, 1);
		this.states = states;
		this.layer = layer;
		
		inputs = LSTMDecoderState.getInputs(states);
		outputs = LSTMDecoderState.getOutputs(states);
	}
	
	public LSTMMapping(LSTMDecoderState[] states, DenseNeuronArray initialState, DenseNeuronArray initialCell,
			LSTMStateTransform layer) {
		super(0, 1, 0, 1);
		this.states = states;
		this.layer = layer;
		this.initialCell = initialCell;
		this.initialState = initialState;
		
		inputs = LSTMDecoderState.getInputs(states);
		
		outputs = LSTMDecoderState.getOutputs(states);
	}
	
	public LSTMMapping(LSTMDecoderState[] states, DenseNeuronArray[] sources, DenseNeuronArray initialState, DenseNeuronArray initialCell,
			LSTMStateTransform layer) {
		super(0, 1, 0, 1);
		this.states = states;
		this.layer = layer;
		this.initialCell = initialCell;
		this.initialState = initialState;
		this.sources = sources;
		inputs = LSTMDecoderState.getInputs(states);
		outputs = LSTMDecoderState.getStates(states);
	}

	@Override
	public void forward() {
		long time = System.currentTimeMillis();
		try{
			layer.forward(this);
		} catch (ArrayIndexOutOfBoundsException e){
			System.err.println("error mapping using layer " + layer);
			throw (e);
		}
		((Layer)layer).addForward(System.currentTimeMillis()-time);
	}

	@Override
	public void backward() {
		long time = System.currentTimeMillis();
		layer.backward(this);
		((Layer)layer).addBackward(System.currentTimeMillis()-time);
	}

	@Override
	public Layer getLayer() {
		return (Layer) layer;
	}

	@Override
	public NeuronArray getInput() {
		return null;
	}

	@Override
	public NeuronArray[] getInputArray() {
		if(sources!=null){
			NeuronArray[] ret = new NeuronArray[inputs.length + sources.length];
			for(int i = 0; i < inputs.length;i++){
				ret[i] = inputs[i];
			}
			for(int i = 0; i < sources.length;i++){
				ret[i+inputs.length] = sources[i];
			}
			return ret;
		}
		return inputs;
	}

	@Override
	public NeuronArray getOutput() {
		return null;
	}

	@Override
	public NeuronArray[] getOutputArray() {
		return outputs;
	}

}
