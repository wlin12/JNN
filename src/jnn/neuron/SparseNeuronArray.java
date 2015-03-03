package jnn.neuron;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;

public class SparseNeuronArray extends NeuronArray{
	public HashMap<Integer, Double> outputs;
	public HashMap<Integer, Double> error;
	public String name;
		
	public SparseNeuronArray(int size) {
		super(size);
		outputs = new HashMap<Integer, Double>();
		error = new HashMap<Integer, Double>();
	}
	
	@Override
	public void init() {
		for(Entry<Integer, Double> el : outputs.entrySet()){
			el.setValue(0.0);
		}
		for(Entry<Integer, Double> el : error.entrySet()){
			el.setValue(0.0);
		}
	}

	public double getError(int index){
		return error.get(index);
	}
	
	public void setWordWindow(int vocabSize, int[] words){
		for(int i = 0; i < words.length; i++){
			outputs.put(i*vocabSize+words[i], 1.0d);
			error.put(i*vocabSize+words[i], 0.0d);
		}
	}

	public void addNeuron(int actId){
		addNeuron(actId, 0);
	}

	public void addNeuron(int actId, double i){
		if(!outputs.containsKey(actId)){
			setNeuron(actId, i);
		}
		else{
			outputs.put(actId, i + outputs.get(actId));
		}
	}
	
	public void setNeuron(int actId, double i) {
		outputs.put(actId, i);
		error.put(actId, 0.0);
	}

		
	public Set<Integer> getNonZeroKeys(){
		return outputs.keySet();
	}
	
	public Set<Entry<Integer, Double>> getNonZeroEntries(){
		return outputs.entrySet();
	}
	
	public int maxIndex(){
		double max = -Double.MAX_VALUE;
		int index = -1;
		for(Entry<Integer, Double> el : outputs.entrySet()){
			if(el.getValue() > max){
				max = el.getValue();
				index = el.getKey();
			}
		}
		return index;
	}
	
	public Set<Entry<Integer, Double>> getNonZeroEntries(int start, int end){	
		if(outputs == null){
			throw new RuntimeException("fecthing outputs for unitialized input " + name);
		}
		HashSet<Entry<Integer, Double>> set = new HashSet<Entry<Integer,Double>>();
		for(Entry<Integer, Double> el : outputs.entrySet()){
			if(el.getKey()>=start && el.getKey() <= end){
				set.add(el);
			}
		}
		return set;
	}
	
	public void addError(int key, double errorToAdd){
		if(!error.containsKey(key)){
			throw new RuntimeException("error key " + key + " doesnt exist");
		}
		double current = error.get(key);
		error.put(key, errorToAdd + current);
	}
	
	@Override
	public String toString() {
		String ret = "";
		if(name != null){
			ret+=name+="\n";
		}
		for(Entry<Integer, Double> el : outputs.entrySet()){
			ret+=el + " ";
		}
		ret+="\n";
		for(Entry<Integer, Double> el : error.entrySet()){
			ret+=el + " ";
		}
		return ret;
	}
	
	public void link(SparseNeuronArray in){
		this.size = in.size;
		this.outputs = in.outputs;
		this.error = in.error;
	}

	public double getOutput(int key) {
		return outputs.get(key);
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public String getName() {
		return name;
	}
	
	@Override
	public void capValues() {
		for(Entry<Integer, Double> el : outputs.entrySet()){
			if(el.getValue()>50){
				el.setValue(50d);
			}
		}
	}
	
	public static String vectorToString(SparseNeuronArray[] vector){
		String ret = "";
		for(int i = 0; i < vector.length; i++){
			ret+=vector[i] + "\n";
		}
		return ret;
	}
}
