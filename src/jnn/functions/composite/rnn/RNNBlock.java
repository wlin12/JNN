package jnn.functions.composite.rnn;

import jnn.functions.nonparametrized.LogisticSigmoidLayer;
import jnn.functions.parametrized.CopyLayer;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.training.TreeInference;

public class RNNBlock {
	DenseNeuronArray hprevState;

	DenseNeuronArray hState;

	int start;
	int end;

	public RNNBlock(DenseNeuronArray hprevState, int start) {
		super();
		this.hprevState = hprevState;
		this.start = start;
	}

	public RNNBlock nextState(){
		return new RNNBlock(hState, end+1);
	}

	public void addToInference(TreeInference inference, DenseNeuronArray inputX, RNNParameters parameters){
		int units = inputX.len();
		int stateSize = hprevState.size;

		int level = start;
		
		DenseNeuronArray stateInput = new DenseNeuronArray(units + stateSize);
		inference.addNeurons(level, stateInput);
		
		inference.addMapping(new OutputMappingDenseToDense(0,units-1,0,units-1,inputX, stateInput, CopyLayer.singleton));
		inference.addMapping(new OutputMappingDenseToDense(0,stateSize-1,units,units+stateSize-1,hprevState, stateInput, CopyLayer.singleton));
		level++;

		DenseNeuronArray stateOutput = new DenseNeuronArray(stateSize);
		inference.addNeurons(level, stateOutput);
		
		inference.addMapping(new OutputMappingDenseToDense(stateInput, stateOutput, parameters.inputTransformLayer));
		
		//next state computation
		level++;
		hState = new DenseNeuronArray(stateSize);
		hState.setName("state at level " + start);
		inference.addNeurons(level, hState);
		inference.addMapping(new OutputMappingDenseToDense(stateOutput , hState, LogisticSigmoidLayer.singleton));
	
		end = level+1;
	}
}