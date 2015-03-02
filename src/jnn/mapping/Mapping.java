package jnn.mapping;

import java.util.HashMap;
import java.util.Random;

import jnn.functions.parametrized.Layer;
import jnn.neuron.NeuronArray;
import jnn.training.TreeInference;
import util.DropoutMatrixGen;
import util.XorRandom;

abstract public class Mapping {
	public static Random rand = new XorRandom(10); 
	
	int inputStart;
	int inputEnd;
	int outputStart;
	int outputEnd;

	boolean dropout = false;
	double dropoutRate = 0;
	boolean[][] droppedOut;
	
	TreeInference parentInference;
	TreeInference subInference;
	HashMap<String, Object> forwardParams = new HashMap<String, Object>();

	Layer timedLayer = null;
	
	public Mapping(int inputStart, int inputEnd, int outputStart, int outputEnd) {
		super();
		this.inputStart = inputStart;
		this.inputEnd = inputEnd;
		this.outputStart = outputStart;
		this.outputEnd = outputEnd;		
	}

	public void validate(){
		if(getInput().size <= inputEnd){
			System.err.println("input was " + getInput());
			throw new RuntimeException("mapping from non-existing input " + inputEnd + " to " + getInput().size);
		}
		if(getOutput().size <= outputEnd){
			System.err.println("output was " + getOutput());
			throw new RuntimeException("mapping to non-existing output " + outputEnd + " to " + getOutput().size);
		}
	}

	public void setDropout(double dropoutRate){
		if(dropoutRate > 0 && isTrain()){
			this.dropout = true;
			this.dropoutRate = dropoutRate;
			this.droppedOut = DropoutMatrixGen.gen(inputEnd-inputStart+1, outputEnd-outputStart+1, dropoutRate);
		}
	}
	
	public boolean isTrain(){
		return parentInference.isTrain();
	}
	
	public boolean useDropout(){
		return dropout;
	}
	
	public double getDropoutRate() {
		return dropoutRate;
	}
	
	public boolean[][] getDroppedOut() {
		return droppedOut;
	}
	
	public int getId() {
		return parentInference.getId();
	}
	
	public TreeInference getSubInference(){
		if(subInference == null){
			this.subInference = new TreeInference(getId());
			this.subInference.setTrain(parentInference.isTrain());			
		}
		return subInference;
	}
	
	public void setParentInference(TreeInference parentInference) {
		this.parentInference = parentInference;
	}
	
	public Object getForwardParam(String key){
		return forwardParams.get(key);
	}
	
	public void setForwardParam(String key, Object obj){
		forwardParams.put(key, obj);
	}
	
	public void timedForward(){
		if(timedLayer!=null){
			long start = System.currentTimeMillis();
			forward();
			timedLayer.addForward(System.currentTimeMillis()-start);
		}
		else{
			forward();
		}
	}
	
	public void timedBackward(){
		if(timedLayer!=null){
			long start = System.currentTimeMillis();
			backward();
			timedLayer.addBackward(System.currentTimeMillis()-start);
		}
		else{
			backward();
		}		
	}
	
	public void setTimedLayer(Object layer){
		timedLayer = (Layer)layer;
	}
	
	abstract public void forward();

	abstract public void backward();

	abstract public Layer getLayer();

	abstract public NeuronArray getInput();

	abstract public NeuronArray[] getInputArray();

	abstract public NeuronArray getOutput();

	abstract public NeuronArray[] getOutputArray();
}
