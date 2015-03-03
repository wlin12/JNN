package jnn.training;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import jnn.functions.SparseToDenseTransform;
import jnn.functions.parametrized.Layer;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingSparseToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;
import jnn.neuron.SparseNeuronArray;

public class TreeInference {
	HashMap<Integer, ArrayList<Mapping>> mappingsPerTargetLevel = new HashMap<Integer, ArrayList<Mapping>>();
	HashMap<Integer, ArrayList<Mapping>> mappingsPerSourceLevel = new HashMap<Integer, ArrayList<Mapping>>();

	HashMap<Integer, ArrayList<NeuronArray>> outputsPerLevel = new HashMap<Integer, ArrayList<NeuronArray>>();
	HashMap<NeuronArray, Integer> neuronsToLevel;

	HashSet<Layer> layers;
	int maxLevel = 0;
	int id = 0;

	// network parameters
	boolean train;

	public TreeInference(int id) {
		this.id = id;
		neuronsToLevel = new HashMap<NeuronArray, Integer>();
		layers = new HashSet<Layer>();
	}

	public void clear(){
		neuronsToLevel.clear();;
		layers.clear();;
		mappingsPerTargetLevel.clear();;
		mappingsPerSourceLevel.clear();;
		outputsPerLevel.clear();;
		maxLevel = 0;
	}

	public void addMapping(SparseNeuronArray sparseNeuronArray,
			DenseNeuronArray denseNeuronArray, SparseToDenseTransform layer) {
		addMapping(new OutputMappingSparseToDense(sparseNeuronArray, denseNeuronArray, layer));
	}

	public void addMapping(Mapping map){
		NeuronArray[] inputArray = map.getInputArray();
		NeuronArray[] outputArray = map.getOutputArray();

		int inputLevel = 0;
		if(inputArray != null){
			for(NeuronArray input : inputArray){
				if(input!= null && !neuronsToLevel.containsKey(input)){
					throw new RuntimeException("input neurons does do not exist for mapping ");			
				}
				if(input!=null){
					inputLevel = Math.max(neuronsToLevel.get(input), inputLevel); 
				}			
			}
		}

		int outputLevel = Integer.MAX_VALUE;
		for(NeuronArray output : outputArray){
			if(!neuronsToLevel.containsKey(output)){
				throw new RuntimeException("output neurons does do not exist for mapping ");
			}
			if(output!=null){
				outputLevel = Math.min(neuronsToLevel.get(output), outputLevel);
			}
		}
		addMapping(inputLevel, outputLevel,map);
	}


	public void addMapping(int sourceLevel, int targetLevel, Mapping map){
		if(targetLevel == 0){
			throw new RuntimeException("no mapping should end in level 0");
		}
		map.setParentInference(this);
		layers.add(map.getLayer());
		if(!mappingsPerSourceLevel.containsKey(sourceLevel)){
			mappingsPerSourceLevel.put(sourceLevel, new ArrayList<Mapping>());
		}
		if(!mappingsPerTargetLevel.containsKey(targetLevel)){
			mappingsPerTargetLevel.put(targetLevel, new ArrayList<Mapping>());
		}

		mappingsPerTargetLevel.get(targetLevel).add(map);
		mappingsPerSourceLevel.get(sourceLevel).add(map);	
	}

	public void addNeurons(NeuronArray neurons){
		addNeurons(getMaxLevel()+1, neurons);
	}

	public void addNeurons(DenseNeuronArray[] neurons){
		int level = getMaxLevel() + 1;
		for(DenseNeuronArray n : neurons){			
			addNeurons(level, n);
		}
	}

	public void addNeurons(int level, NeuronArray neurons){
		if(neuronsToLevel.containsKey(neurons)){
			throw new RuntimeException("tree already contains neurons " + neurons);
		}
		if(!outputsPerLevel.containsKey(level)){
			outputsPerLevel.put(level, new ArrayList<NeuronArray>());
		}
		outputsPerLevel.get(level).add(neurons);
		neuronsToLevel.put(neurons, level);
		if(maxLevel < level){
			maxLevel = level;
		}
	}

	public void init(){			
		for(Entry<Integer, ArrayList<NeuronArray>> entry : outputsPerLevel.entrySet()){
			if(entry.getKey() == 0) continue;
			for(NeuronArray neurons : entry.getValue()){
				neurons.init();
			}
		}
	}

	public long forward(){
		long time = System.currentTimeMillis();
		for(int i = 0; i <= maxLevel; i++){
			if(mappingsPerSourceLevel.containsKey(i)){
				forwardLevel(i);			
			}
		}
		return System.currentTimeMillis() - time;
	}

	public void forwardLevel(int level){
		if(outputsPerLevel.containsKey(level)){
			for(NeuronArray neurons : outputsPerLevel.get(level)){
				neurons.beforeForward();
			}
		}
		for(Mapping map : mappingsPerSourceLevel.get(level)){
			map.timedForward();
		}
	}

	public long backward(){
		long time = System.currentTimeMillis();
		for(int i = maxLevel; i > 0; i--){
			if(mappingsPerTargetLevel.containsKey(i)){				
				backwardLevel(i);			
			}
		}
		return System.currentTimeMillis() - time;
	}

	public void backwardLevel(int level){
		for(NeuronArray neurons : outputsPerLevel.get(level)){
			neurons.beforeBackward();
		}
		for(Mapping map : mappingsPerTargetLevel.get(level)){
			map.timedBackward();
		}
	}

	public long commit(double learningRate){
		long time = System.currentTimeMillis();
		for(Layer layer : layers){
			layer.updateWeightsTimed(learningRate, 0);
		}		
		return System.currentTimeMillis() - time;
	}	

	public int getNeuronLayer(NeuronArray neurons){
		return neuronsToLevel.get(neurons);
	}

	public void printNeurons(){
		for(Entry<Integer, ArrayList<NeuronArray>> entry : outputsPerLevel.entrySet()){
			System.err.println("neurons for level " + entry.getKey());
			for(NeuronArray neurons : entry.getValue()){
				System.err.println(neurons);
			}
		}		
	}

	public void printMappings(){
		for(Entry<Integer, ArrayList<Mapping>> entry : mappingsPerSourceLevel.entrySet()){
			System.err.println("mapping for level " + entry.getKey());
			for(Mapping map : entry.getValue()){
				System.err.println(map);
			}
		}
	}

	public void freeLayers(){
		for(Layer layer : layers){
			layer.reset();
		}
	}	

	public int getMaxLevel() {
		return maxLevel;
	}

	public boolean isTrain(){
		return train;
	}

	public void setTrain(boolean train) {
		this.train = train;
	}

	public int getId() {
		return id;
	}

	public void setId(int id) {
		this.id = id;
	}
}
