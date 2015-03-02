package jnn.features;

import java.util.HashMap;

import org.nd4j.linalg.api.ndarray.INDArray;

import util.MathUtils;

public class GradientStore {
	HashMap<Integer, INDArray> gradientsPerProcess = new HashMap<Integer, INDArray>();
	HashMap<Integer, Integer> updatesPerProcess = new HashMap<Integer, Integer>();
	public static boolean cropGradients = true;
	
	public void init(){
		gradientsPerProcess.clear();
		updatesPerProcess.clear();
	}

	public void addGradient(int processId, INDArray gradient){
		if(!checkEmptyAndAddForGradient(processId, gradient)){
			gradientsPerProcess.get(processId).addi(gradient);
			updatesPerProcess.put(processId, updatesPerProcess.get(processId) + 1);
		}
	}	

	public synchronized boolean checkEmptyAndAddForGradient(int processId, INDArray gradient){
		if(!gradientsPerProcess.containsKey(processId)){
			gradientsPerProcess.put(processId, gradient);
			updatesPerProcess.put(processId, 1);
			return true;
		}
		return false;
	}

	public INDArray getGradientAvg(){
		if(gradientsPerProcess.isEmpty()){
			return null;
		}
		INDArray ret = null;
		int sum = 0;
		for(int processId : gradientsPerProcess.keySet()){
			if(ret == null){
				ret = gradientsPerProcess.get(processId);
			}
			else{
				ret.addi(gradientsPerProcess.get(processId));
			}
			sum+=updatesPerProcess.get(processId);
		}
		ret = ret.div(sum);
		if(cropGradients){
			MathUtils.cap(ret, 10);
		}
		return ret;
	}

	public INDArray getGradientSum(){
		if(gradientsPerProcess.isEmpty()){
			return null;
		}
		INDArray ret = null;
		for(int processId : gradientsPerProcess.keySet()){
			if(ret == null){
				ret = gradientsPerProcess.get(processId);
			}
			else{
				ret.addi(gradientsPerProcess.get(processId));
			}
		}
		if(ret !=null && cropGradients){
			MathUtils.cap(ret, 10);
		}
		return ret;
	}
}
