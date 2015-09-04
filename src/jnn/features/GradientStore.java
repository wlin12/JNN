package jnn.features;

import java.util.HashMap;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import util.MathUtils;

public class GradientStore {

	HashMap<Integer, INDArray> gradientsPerProcess = new HashMap<Integer, INDArray>();
	
	private int maxInputSize = 100;
	HashMap<Integer, INDArray> outputGradsPerProcess = new HashMap<Integer, INDArray>();
	HashMap<Integer, INDArray> inputsPerProcess = new HashMap<Integer, INDArray>();
	HashMap<Integer, Integer> inputsInCache = new HashMap<Integer, Integer>();

	HashMap<Integer, Integer> updatesPerProcess = new HashMap<Integer, Integer>();
	
	public static boolean cropGradients = true;

	public void init(){
		gradientsPerProcess.clear();
		updatesPerProcess.clear();
		inputsPerProcess.clear();
		outputGradsPerProcess.clear();
		inputsInCache.clear();
	}

	public void addGradient(int processId, INDArray gradient){
		if(!checkEmptyAndAddForGradient(processId, gradient)){
			gradientsPerProcess.get(processId).addi(gradient);
			updatesPerProcess.put(processId, updatesPerProcess.get(processId) + 1);
		}
	}

	public void computeGradientAndAdd(){
		int numberOfThreads = inputsPerProcess.keySet().size();
		Integer[] keys = inputsPerProcess.keySet().toArray(new Integer[]{});
		Thread[] thread = new Thread[numberOfThreads];
		
		for(int i = 0; i < numberOfThreads; i++){
			final int id = keys[i];
			thread[i] = new Thread(){
				@Override
				public void run() {
					computeGradientAndAdd(id);
				}
			};
			thread[i].start();
		}
		for(int i = 0; i < numberOfThreads; i++){
			try {
				thread[i].join();
			} catch (InterruptedException e) {
				throw new RuntimeException(e);
			}
		}
	}
	
	public void computeGradientAndAdd(int processId){
		int numberOfElements = inputsInCache.get(processId);
		if(numberOfElements >0){
			INDArray x = inputsPerProcess.get(processId);
			INDArray yGrad = outputGradsPerProcess.get(processId);
			
			int[] rows = new int[numberOfElements];
			for(int i = 0; i < numberOfElements; i++){
				rows[i] = i;
			}
			INDArray xCut = x.getRows(rows);
			INDArray yGradCut = yGrad.getRows(rows);
			INDArray wGrad = xCut.transpose().mmul(yGradCut);
			
			addGradient(processId, wGrad);
			x.muli(0);
			yGrad.muli(0);
			inputsInCache.put(processId, 0);
		}
	}

	public void addInputAndOutput(int processId, INDArray input, INDArray output){
		if(!checkEmptyAndAddForInputAndOutput(processId, input, output)){
			int numberOfElements = inputsInCache.get(processId);
			inputsPerProcess.get(processId).putRow(numberOfElements,input);
			outputGradsPerProcess.get(processId).putRow(numberOfElements,output);
			updatesPerProcess.put(processId, updatesPerProcess.get(processId) + 1);
			numberOfElements++;
			inputsInCache.put(processId, numberOfElements);
			if(numberOfElements == maxInputSize){
				computeGradientAndAdd(processId);
			}
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

	public synchronized boolean checkEmptyAndAddForInputAndOutput(int processId, INDArray input, INDArray output){
		if(!inputsPerProcess.containsKey(processId)){
			inputsPerProcess.put(processId, Nd4j.zeros(new int[]{maxInputSize, input.size(0)}));
			outputGradsPerProcess.put(processId, Nd4j.zeros(new int[]{maxInputSize, output.size(0)}));
			inputsPerProcess.get(processId).getRow(0).addi(input);
			outputGradsPerProcess.get(processId).getRow(0).addi(output);
			inputsInCache.put(processId,1);
			updatesPerProcess.put(processId, 1);
			return true;
		}
		return false;
	}

	public INDArray getGradientAvg(){
		if(inputsInCache.size() != 0){
			computeGradientAndAdd();
		}
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
		if(inputsInCache.size() != 0){
			computeGradientAndAdd();
		}
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