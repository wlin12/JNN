package jnn.functions.composite.lstm.aux;

import jnn.functions.nonparametrized.CopyLayer;
import jnn.functions.nonparametrized.LogisticSigmoidLayer;
import jnn.functions.nonparametrized.TanSigmoidLayer;
import jnn.functions.parametrized.HadamardProductLayer;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.training.GraphInference;

public class LSTMBlock {
	public DenseNeuronArray hprevState;

	public DenseNeuronArray iGate;
	public DenseNeuronArray fGate;
	public DenseNeuronArray oGate;
	public DenseNeuronArray cGate;

	public DenseNeuronArray cprevMem;
	public DenseNeuronArray cMen;

	public DenseNeuronArray hState;

	public int start;
	public int end;

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

	public void addToInference(GraphInference inference, DenseNeuronArray inputX, LSTMParameters parameters){
		int units = inputX.len();
		if(units == 0){
			throw new RuntimeException("input has size 0");
		}
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
	
	public LSTMBlock[] addMultipleBlocks(GraphInference inference, DenseNeuronArray[] input, LSTMParameters parameters){
		LSTMBlock[] blocks = new LSTMBlock[input.length];
		LSTMBlock currentBlock = this;
		for(int i = 0; i < input.length; i++){
			currentBlock.addToInference(inference, input[i], parameters);
			blocks[i] = currentBlock;
			currentBlock = currentBlock.nextState();
		}
		return blocks;
	}
	
	public LSTMBlock[] addMultipleBlocksReverse(GraphInference inference, DenseNeuronArray[] input, LSTMParameters parameters){
		LSTMBlock[] blocks = new LSTMBlock[input.length];
		LSTMBlock currentBlock = this;
		for(int i = 0; i < input.length; i++){
			currentBlock.addToInference(inference, input[input.length-i-1], parameters);
			blocks[input.length-i-1] = currentBlock;
			currentBlock = currentBlock.nextState();
		}
		return blocks;
	}
}
