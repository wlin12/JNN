package jnn.training;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Scanner;
import java.util.TreeMap;

import org.apache.commons.math3.util.FastMath;

import util.MapUtils;
import util.TopologicalSort;
import jnn.functions.SparseToDenseTransform;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingSparseToDense;
import jnn.neuron.CompositeNeuronArray;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.NeuronArray;
import jnn.neuron.SparseNeuronArray;

public class GraphInference {
	public static int INPUT_LEVEL = 0;
	public static int OUTPUT_LEVEL = Integer.MAX_VALUE;

	HashMap<Integer, Mapping> idToMappings = new HashMap<Integer, Mapping>();
	HashMap<Mapping, Integer> mappingsToId = new HashMap<Mapping, Integer>();
	HashMap<Integer, NeuronArray> idToNeurons = new HashMap<Integer, NeuronArray>();
	HashMap<NeuronArray, Integer> neuronsToId = new HashMap<NeuronArray, Integer>();
	HashSet<Layer> layers = new HashSet<Layer>();
	HashMap<Integer, LinkedList<Mapping>> inputNeuronsToMappings = new HashMap<Integer, LinkedList<Mapping>>();
	HashMap<Integer, LinkedList<Mapping>> outputNeuronsToMappings = new HashMap<Integer, LinkedList<Mapping>>();

	int id = 0;
	double norm = 1;

	// network parameters
	boolean train;
	boolean validate = false;

	// forward results
	List<List<Mapping>> forwardOrder;
	List<List<Mapping>> backwardOrder;

	long inferenceTime = 0;
	long forwardTime = 0;
	long backwardTime = 0;
	
	// debug
	int numberOfRepeatedNeurons = 0;

	public GraphInference(int id, boolean train) {
		this.id = id;
		this.train = train;
	}

	public void clear(){
		layers.clear();
		idToNeurons.clear();
		layers.clear();
		neuronsToId.clear();
		mappingsToId.clear();
		idToMappings.clear();
		outputNeuronsToMappings.clear();
		inputNeuronsToMappings.clear();
		forwardOrder = null;
		backwardOrder = null;
	}

	public void addMapping(SparseNeuronArray sparseNeuronArray,
			DenseNeuronArray denseNeuronArray, SparseToDenseTransform layer) {
		addMapping(new OutputMappingSparseToDense(sparseNeuronArray, denseNeuronArray, layer));
	}

	public void addMapping(Mapping map){
		NeuronArray[] inputArray = map.getInputArray();
		NeuronArray[] outputArray = map.getOutputArray();

		if(inputArray != null){
			for(NeuronArray input : inputArray){				
				if(input!= null && !neuronsToId.containsKey(input)){
					throw new RuntimeException("input neurons does do not exist for mapping ");			
				}
				if(input!=null){
					int neuronId = neuronsToId.get(input);
					inputNeuronsToMappings.get(neuronId).add(map);
				}			
			}
		}

		if(outputArray != null){
			for(NeuronArray output : outputArray){
				if(!neuronsToId.containsKey(output)){
					throw new RuntimeException("output neurons does do not exist for mapping ");
				}
				if(output!=null){
					int neuronId = neuronsToId.get(output);
					outputNeuronsToMappings.get(neuronId).add(map);
				}
			}
		}
		int mappingId = idToMappings.size();
		idToMappings.put(mappingId, map);
		mappingsToId.put(map, mappingId);
		layers.add(map.getLayer());
		map.setParentInference(this);
	}

	public void addNeurons(CompositeNeuronArray neurons){
		addNeurons(neurons.getAtomicNeurons());
	}

	public void addNeurons(int level, CompositeNeuronArray neurons){
		addNeurons(neurons.getAtomicNeurons());
	}

	public void addNeurons(int level, NeuronArray[] neurons){
		for(NeuronArray n : neurons){			
			addNeurons(n);
		}
	}

	public void addNeurons(NeuronArray[] neurons){
		for(NeuronArray n : neurons){			
			addNeurons(n);
		}
	}

	public void addNeurons(int level, NeuronArray neurons){
		addNeurons(neurons);
	}

	public void addNeurons(NeuronArray neurons){
		if(neuronsToId.containsKey(neurons)){
			//			numberOfRepeatedNeurons++;
			//			return;
			throw new RuntimeException("tree already contains neurons " + neurons);
		}
		int neuronId = neuronsToId.size();
		neuronsToId.put(neurons, neuronId);
		idToNeurons.put(neuronId, neurons);
		inputNeuronsToMappings.put(neuronId, new LinkedList<Mapping>());
		outputNeuronsToMappings.put(neuronId, new LinkedList<Mapping>());
	}

	public void init(){
		for(int id = 0 ; id < neuronsToId.size(); id++){
			boolean isInput = outputNeuronsToMappings.get(id).size() == 0;
			if(!isInput){
				idToNeurons.get(id).init();
			}
		}
	}

	private List<List<Integer>> forwardTopologicalOrder(){
		LinkedList<Integer>[] graph = new LinkedList[neuronsToId.size()];
		for(int id = 0; id < graph.length; id++){
			graph[id] = new LinkedList<Integer>();
			for(Mapping map : inputNeuronsToMappings.get(id)){
				NeuronArray[] outputs = map.getOutputArray();
				if(outputs!=null){
					for(NeuronArray output : outputs){
						if(output != null){
							graph[id].add(neuronsToId.get(output));
						}
					}
				}
			}
		}
		return TopologicalSort.blockTopologicalSort(graph);
	}

	private List<List<Integer>> backwardTopologicalOrder(){
		LinkedList<Integer>[] graph = new LinkedList[neuronsToId.size()];
		for(int id = 0; id < graph.length; id++){
			graph[id] = new LinkedList<Integer>();
			for(Mapping map : outputNeuronsToMappings.get(id)){
				NeuronArray[] inputs = map.getInputArray();
				if(inputs!=null){
					for(NeuronArray input : inputs){
						if(input != null){
							graph[id].add(neuronsToId.get(input));
						}
					}
				}
			}
		}
		return TopologicalSort.blockTopologicalSort(graph);
	}

	private List<List<Mapping>> forwardMappingBlocks(){
		List<List<Integer>> forwardTopological = forwardTopologicalOrder();
		int[] lastTimestampToExpand = new int[idToMappings.size()];
		int[] firstTimestampToExpand = new int[idToMappings.size()];
		for(int i = 0; i < lastTimestampToExpand.length; i++){
			lastTimestampToExpand[i] = forwardTopological.size();
			firstTimestampToExpand[i] = 0;
		}
		int timestamp = 0;
		for(List<Integer> inputsForTimestamp : forwardTopological){
			for(int id : inputsForTimestamp){
				LinkedList<Mapping> mappingsThatMustForwardAfter = inputNeuronsToMappings.get(id);
				for(Mapping mapThatMustForwardAfter : mappingsThatMustForwardAfter){
					int mapId = mappingsToId.get(mapThatMustForwardAfter);
					firstTimestampToExpand[mapId] = FastMath.max(firstTimestampToExpand[mapId], timestamp);
				}			

				LinkedList<Mapping> mappingsThatMustForwardBefore = outputNeuronsToMappings.get(id);			
				for(Mapping mapThatMustForwardBefore : mappingsThatMustForwardBefore){
					int mapId = mappingsToId.get(mapThatMustForwardBefore);
					lastTimestampToExpand[mapId] = FastMath.min(lastTimestampToExpand[mapId], timestamp-1);				
				}
			}
			timestamp++;
		}

		TreeMap<Integer, List<Mapping>> mappingsPerTimestamp = new TreeMap<Integer, List<Mapping>>();
		for(int mappingId = 0; mappingId < idToMappings.size(); mappingId++){
			int t = firstTimestampToExpand[mappingId];
			if(!mappingsPerTimestamp.containsKey(t)){
				mappingsPerTimestamp.put(t, new LinkedList<Mapping>());
			}
			mappingsPerTimestamp.get(t).add(idToMappings.get(mappingId));
		}
		List<List<Mapping>> ret = new LinkedList<List<Mapping>>();
		for(int t : mappingsPerTimestamp.navigableKeySet()){
			for(Mapping map : mappingsPerTimestamp.get(t)){
				LinkedList<Mapping> block = new LinkedList<Mapping>();
				block.add(map);
				ret.add(block);
			}
		}
		return ret;
	}

	public long forward(){
		long time = System.currentTimeMillis();

		if(forwardOrder!=null){
			throw new RuntimeException("already ran forward");
		}
		long topologicalOrderStart = System.currentTimeMillis();
		forwardOrder = forwardMappingBlocks();
		inferenceTime += System.currentTimeMillis() - topologicalOrderStart;
		
		if(validate){
			boolean[] usedAsInput = new boolean[idToNeurons.size()];
			boolean[] usedAsOutput = new boolean[idToNeurons.size()];
			for(List<Mapping> block : forwardOrder){
				if(block.size() == 1){
					Mapping map = block.get(0);
					NeuronArray[] inputs = map.getInputArray();
					if(inputs != null){
						for(int i = 0; i < inputs.length; i++){
							if(inputs[i]!=null){
								NeuronArray neurons = inputs[i];
								int neuronsId = neuronsToId.get(neurons);
								if(!usedAsInput[neuronsId]){
									usedAsInput[neuronsId] = true;
								}
							}
						}
					}
					NeuronArray[] outputs = map.getOutputArray();
					if(outputs != null){
						for(int o = 0; o < outputs.length; o++){
							if(outputs[o]!=null){
								NeuronArray neurons = outputs[o];
								int neuronsId = neuronsToId.get(neurons);
								if(!usedAsOutput[neuronsId]){
									usedAsOutput[neuronsId] = true;
								}
								if(usedAsInput[neuronsId]){
									throw new RuntimeException("outputing to already used input\n"+neurons);
								}
							}
						}
					}
				}
			}
		}
		for(List<Mapping> block : forwardOrder){
			if(block.size() == 1){
				block.get(0).timedForward();
			}
		}
		long totalTime = System.currentTimeMillis() - time;
		forwardTime += totalTime;
		return totalTime;
	}

	public long backward(){

		
		long time = System.currentTimeMillis();
		if(backwardOrder==null){
			backwardOrder = new LinkedList<List<Mapping>>();
			backwardOrder.addAll(forwardOrder);
		}

		Collections.reverse(backwardOrder);
		inferenceTime+=System.currentTimeMillis() - time;
		
		time = System.currentTimeMillis();
		for(List<Mapping> block : backwardOrder){
			if(block.size() == 1){
				block.get(0).timedBackward();
			}
		}
		long totalTime = System.currentTimeMillis() - time;
		backwardTime += totalTime;
		return totalTime;	
	}

	public void checkin(){
		for(Layer l : layers ){
			l.backwardEnd(this);
		}
	}

	public long commit(double learningRate){
		long time = System.currentTimeMillis();
		for(Layer layer : layers){
			layer.updateWeightsTimed(learningRate, 0);
		}		
		return System.currentTimeMillis() - time;
	}	

	public void printNeurons(){
		if(forwardOrder==null){
			forwardOrder = forwardMappingBlocks();
		}
		int order = 0;
		boolean[] used = new boolean[idToNeurons.size()];
		for(List<Mapping> block : forwardOrder){
			for(Mapping map : block){
				NeuronArray[] inputs = map.getInputArray();
				if(inputs != null){
					for(int i = 0; i < inputs.length; i++){
						if(inputs[i]!=null){
							NeuronArray neurons = inputs[i];
							int neuronsId = neuronsToId.get(neurons);
							if(!used[neuronsId]){
								used[neuronsId] = true;
								System.err.println("neurons-" + (++order));
								System.err.println(neurons);
							}
						}
					}
				}
				NeuronArray[] outputs = map.getOutputArray();
				if(outputs != null){
					for(int o = 0; o < outputs.length; o++){
						if(outputs[o]!=null){
							NeuronArray neurons = outputs[o];
							int neuronsId = neuronsToId.get(neurons);
							if(!used[neuronsId]){
								used[neuronsId] = true;
								System.err.println("output neurons-" + (++order));
								System.err.println(neurons);
							}
						}
					}
				}
			}
		}
	}

	public void printMappings(){
		printMappings(System.err);
	}
	
	public void printMappings(PrintStream out){
		if(forwardOrder==null){
			forwardOrder = forwardMappingBlocks();
		}
		int order = 0;
		for(List<Mapping> block : forwardOrder){
			if(block.size() == 1){
				out.println("----------------------");
				out.println("mapping " + (++order) + "\n" + block.get(0).toString());
				out.println("----------------------");
				if(block.get(0).containsSubInference()){
					block.get(0).getSubInference().printMappings(out);
				}
			}
		}
	}

	public void freeLayers(){
		for(Layer layer : layers){
			layer.reset();
		}
	}	

	public int getMaxLevel() {
		return 0; // not really needed just to match interface
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

	public void setNorm(double norm) {
		this.norm = norm;
	}

	public double getNorm() {
		return norm;
	}

	//	public void navigate(){
	//		Scanner userInput = new Scanner(System.in);
	//		boolean finish = false;
	//		for(int i : mappingsPerSourceLevel.keySet()){
	//			if(finish){
	//				break;
	//			}
	//			for(Mapping map : mappingsPerSourceLevel.get(i)){
	//				if(finish){
	//					break;
	//				}
	//				NeuronArray[] inputs = null;				
	//				if(map.getInputArray()!=null){
	//					inputs = map.getInputArray();
	//				}
	//				else if (map.getInput()!=null){
	//					inputs = new NeuronArray[]{map.getInput()};
	//				}
	//				if(inputs == null){
	//					System.err.println("no input");
	//				}
	//				else{
	//					System.err.println("input level = " + i);
	//					for(NeuronArray n : inputs){
	//						System.err.println(n);
	//					}
	//				}
	//
	//				NeuronArray[] outputs = null;
	//				if(map.getOutputArray()!=null){
	//					outputs = map.getOutputArray();
	//				}
	//				else if (map.getOutput()!=null){
	//					outputs = new NeuronArray[]{map.getOutput()};
	//				}
	//				if(outputs == null){
	//					System.err.println("no output");
	//				}
	//				else{
	//					System.err.println("output level = " + neuronsToLevel.get(outputs[0]));
	//					for(NeuronArray n : outputs){
	//						System.err.println(n);
	//					}
	//				}
	//
	//				while(true){
	//					System.err.println("what to do? [next][in][out][quit]");
	//					String cmd = userInput.next();
	//					if(cmd.equals("quit")){
	//						System.exit(0);
	//					}
	//					if(cmd.equals("next")||cmd.equals("n")){
	//						break;
	//					}
	//					if(cmd.equals("in")){
	//						map.getSubInference().navigate();
	//						break;
	//					}
	//					if(cmd.equals("out")){
	//						finish = true;
	//						break;
	//					}
	//				}
	//			}
	//		}
	//	}

	public int getNumberOfNeurons() {
		return idToNeurons.size();	
	}

	public GraphInference getSubInference(){

		GraphInference subInference = new GraphInference(getId(), isTrain());
		subInference.setNorm(getNorm());
		subInference.setTrain(isTrain());			

		return subInference;
	}
	
	public long getInferenceTime() {
		long subinferenceTime = 0;
		for(Mapping map : mappingsToId.keySet()){
			if(map.containsSubInference()){
				subinferenceTime+=map.getSubInference().getInferenceTime();
			}
		}
		return inferenceTime + subinferenceTime;
	}
}
