package jnn.functions.composite.lstm;

import java.util.LinkedList;

import jnn.neuron.DenseNeuronArray;

public class LSTMDecoderWithAlignmentState extends LSTMDecoderState{
	public DenseNeuronArray alignments;
	public DenseNeuronArray alignmentsSoftmax;
	
	public LSTMDecoderWithAlignmentState(double score, boolean isFinal,
			DenseNeuronArray output, DenseNeuronArray lstmState,
			DenseNeuronArray lstmCell, DenseNeuronArray input) {
		super(score, isFinal, output, lstmState, lstmCell, input);
	}	
	
	public static LSTMDecoderWithAlignmentState[] buildStateSequence(DenseNeuronArray[] inputs, int stateSize){
		LSTMDecoderWithAlignmentState[] states = new LSTMDecoderWithAlignmentState[inputs.length];
		for(int i = 0; i < states.length; i++){
			boolean isFinal = i == states.length-1;
			DenseNeuronArray lstmState = new DenseNeuronArray(stateSize);
			DenseNeuronArray lstmCell = new DenseNeuronArray(stateSize);
			LSTMDecoderWithAlignmentState state = new LSTMDecoderWithAlignmentState(-1, isFinal, lstmState, lstmState, lstmCell, inputs[i]);
			states[i] = state;
		}
		return states;
	}
	
	public void printAlignments(String[] source, String[] target){
		LSTMDecoderWithAlignmentState currentState = this;
		LinkedList<DenseNeuronArray> alignmentsPerState = new LinkedList<DenseNeuronArray>();
		LinkedList<DenseNeuronArray> alignmentSoftmaxPerState = new LinkedList<DenseNeuronArray>();
		while(currentState != null){
			alignmentsPerState.addFirst(currentState.alignments);
			alignmentSoftmaxPerState.addFirst(currentState.alignmentsSoftmax);
//			System.err.println(currentState.isFinal);
//			System.err.println(currentState.output);
			
			currentState = (LSTMDecoderWithAlignmentState)currentState.prevState;
		}
		System.err.print("source: ");
		for(String s : source) System.err.print(s+ " "); 
		System.err.println();
		for(int i = 0; i < alignmentSoftmaxPerState.size()-1; i++){
			int index = alignmentSoftmaxPerState.get(i).maxIndex();
			String sourceWord = "</s>";
			if(index < source.length){
				sourceWord = source[index];
			}
			String targetWord = "</s>";
			if(i < target.length){
				targetWord = target[i];
			}
			System.err.println(targetWord + "[" + i + "]" + " -> " + sourceWord + "[" + index + "]" + " (" + alignmentSoftmaxPerState.get(i).getNeuron(index) + ")");
//			System.err.println(alignmentsPerState.get(i));		
//			System.err.println(alignmentSoftmaxPerState.get(i));
		}
	}
	
	public double[][] getAlignments(){
		LSTMDecoderWithAlignmentState currentState = this;
		LinkedList<DenseNeuronArray> alignmentSoftmaxPerState = new LinkedList<DenseNeuronArray>();
		while(currentState != null){
			alignmentSoftmaxPerState.addFirst(currentState.alignmentsSoftmax);			
			currentState = (LSTMDecoderWithAlignmentState)currentState.prevState;
		}
		
		double[][] ret = new double[alignmentSoftmaxPerState.get(0).size][alignmentSoftmaxPerState.size()-1];
		for(int i = 0; i < alignmentSoftmaxPerState.size()-1; i++){
			for(int j = 0; j < ret.length; j++){
				ret[j][i] = alignmentSoftmaxPerState.get(i).getNeuron(j);
			}
		}
		return ret;
	}
}
