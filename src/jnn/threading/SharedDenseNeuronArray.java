package jnn.threading;

import java.util.HashSet;

import jnn.neuron.DenseNeuronArray;

public class SharedDenseNeuronArray {
	DenseNeuronArray original;
	
	HashSet<DenseNeuronArray> copies = new HashSet<DenseNeuronArray>();

	public SharedDenseNeuronArray(DenseNeuronArray original) {
		super();
		this.original = original;
	}	
	
	public DenseNeuronArray getCopy(){
		DenseNeuronArray copy = original.copy();
		addCopy(copy);
		return copy;
	}
	
	private synchronized void addCopy(DenseNeuronArray copy){
		copies.add(copy);
	}
	
	public synchronized void mergeCopies(){
		for(DenseNeuronArray sharedCombined : copies){
			original.addError(sharedCombined);
		}
		copies.clear();
	}
	
	public synchronized boolean requiresMerge(){
		return copies.size() > 0;
	}
	
	public DenseNeuronArray getOriginal() {
		return original;
	}
}
