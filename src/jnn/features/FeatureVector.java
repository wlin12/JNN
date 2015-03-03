package jnn.features;

import java.io.PrintStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Scanner;

import util.PrintUtils;
import util.RandomUtils;
import util.SerializeUtils;

public class FeatureVector {


	public static double learningRateDefault = 0.05;
	
	public static boolean useAdagradDefault = false;

	public static boolean useMomentumDefault = false;

	public static boolean useAdadeltaDefault = false;

	public static boolean useAdagradL1Default = false;
	
	public static double l2regularizerLambdaDefault = 0.005;
	
	public static double noiseDevDefault = 0.01;
	
	public static boolean addNoiseDefault = true;

	public static boolean sparseDefault = false;	

	public static int commitMethodDefault = 1;

	public static double maxVal = 10000;
	
	public static double maxError = 1;

	public static double adagradL1LambdaDefault=0.00025;

	public static boolean fastUpdate = false;
	
	public static double momentumDefault = 0.95;

	public static double adadeltaMomentumDefault = 0.95;

	public static double adadeltaEpsDefault = 0.000001;

	public static int initializationType = 0; // 0 -> uniform 0-0.1, 1 -> based on fanin fanout;
	
	int dim;
	double[] features;
	double[] initialization;

	boolean sparse;
	HashSet<Integer> updatedIndexes = new HashSet<Integer>();
	long currentTime;
	long[] timePerFeature;

	long numberOfUpdates = 0 ;

	boolean useMomentum = true;
	double[] prevUpdate;

	boolean useAdagrad = true;
	double[] adagradQuotients;

	double adaGradMax = 100000;
	boolean useAdagradL1 = true;
	double[] avgSubgradient;
	double adagradL1Lambda;

	boolean useAdadelta = false;
	double[] adadeltaRMSGradient;
	double[] adadeltaRMSUpdate;
	double adadeltaMomentum = 0.95;	
	double adadeltaEpsilon = 0.000001;

	boolean useL2 = true;
	boolean useL1 = true;
	public double regL2;

	public boolean commitRequired = true;
	public int commitMethod = 1; // commit = 0 -> average, commit = 1 -> sum, commit = 2 -> norm
	public HashMap<Integer,double[]> gradientsPerFeature = new HashMap<Integer,double[]>();
	public HashMap<Integer,int[]> updatesPerFeature = new HashMap<Integer,int[]>();;


	public FeatureVector(int dim) {
		this(dim, useAdagradL1Default);	
	}

	public FeatureVector(int dim, boolean useAdagradL1) {
		this.dim=dim;
		features = new double[dim];

		this.useAdagrad = useAdagradDefault;
		this.useMomentum = useMomentumDefault;
		this.commitMethod = commitMethodDefault;
		this.useAdadelta = useAdadeltaDefault;
		this.useAdagradL1 = useAdagradL1;
		this.adagradL1Lambda = adagradL1LambdaDefault;
		this.regL2 = l2regularizerLambdaDefault;
		sparse = sparseDefault;
		if(useAdagrad){
			adagradQuotients = new double[dim];
		}
		if(useMomentum){
			prevUpdate = new double[dim];
		}
		if(useAdadelta){
			adadeltaRMSGradient = new double[dim];
			adadeltaRMSUpdate = new double[dim];
		}
		if(useAdagradL1){
			adagradQuotients = new double[dim];
			avgSubgradient = new double[dim];
		}		
		randInitialize();
	}

	public double getRegularizationTarget(int i){
		if(initialization == null) return 0;
		return initialization[i];
	}

	public void setRegularizationTarget(int i, double val){
		if(initialization == null) initialization = new double[dim];
		initialization[i] = val;
	}

	public void fastUpdate(int i, double gradient, double learningRate){
		features[i] += gradient*learningRate;  
	}

	public void update(int i, double gradient, double learningRate, double momentum){
		if(learningRate == 0){
			return;
		}

		double update = 0;
		if(useAdadelta){
			adadeltaRMSGradient[i]=adadeltaRMSGradient[i]*adadeltaMomentum + (1-adadeltaMomentum)*(gradient)*(gradient);
			update = gradient * Math.sqrt(adadeltaRMSUpdate[i]+adadeltaEpsilon) / Math.sqrt(adadeltaRMSGradient[i]+adadeltaEpsilon);
			adadeltaRMSUpdate[i]=adadeltaRMSUpdate[i]*adadeltaMomentum + (1-adadeltaMomentum)*update*update;
		}
		else if(useAdagrad){

			if(useL2){				
				if(sparse){
					long numberOfMissedUpdates = currentTime - timePerFeature[i]; 
					if(numberOfMissedUpdates>0){

						for(int j = 0; j < numberOfMissedUpdates; j++){
							double missedGrad = regL2*(features[i] - getRegularizationTarget(i));
							adagradQuotients[i]+=(missedGrad)*(missedGrad);
							if(adagradQuotients[i]>0){
								if(adagradQuotients[i]>adaGradMax){
									adagradQuotients[i] = adaGradMax;
								}
								double sqrtAdaQ = Math.sqrt(adagradQuotients[i]);								
								features[i] -= regL2*missedGrad*learningRate/sqrtAdaQ;
							}
						}
					}
				}
				double regGradient = gradient - regL2*(features[i] - getRegularizationTarget(i));
				adagradQuotients[i]+=(regGradient)*(regGradient);
				if(adagradQuotients[i]>0){

					if(adagradQuotients[i]>adaGradMax){
						adagradQuotients[i] = adaGradMax;
					}
					double sqrtAdaQ = Math.sqrt(adagradQuotients[i]);
					update = regGradient*learningRate/sqrtAdaQ;
				}
			}
			else{
				adagradQuotients[i]+=(gradient)*(gradient);
				if(adagradQuotients[i]>0){
					if(adagradQuotients[i]>adaGradMax){
						adagradQuotients[i] = adaGradMax;
					}
					double sqrtAdaQ = Math.sqrt(adagradQuotients[i]);
					update= learningRate * gradient / sqrtAdaQ;
				}
			}
		} 
		else if(useAdagradL1){
			adagradQuotients[i]+=(gradient)*(gradient);
			if(adagradQuotients[i]>adaGradMax){
				adagradQuotients[i] = adaGradMax;
			}

			avgSubgradient[i]+=gradient;
			if(adagradQuotients[i] != 0){
				double absSubgrad = Math.abs(avgSubgradient[i])/(numberOfUpdates+1);
				if(absSubgrad<adagradL1Lambda){
					features[i] = 0;
				}
				else{
					features[i] = (absSubgrad-adagradL1Lambda) * learningRate * (numberOfUpdates+1)/Math.sqrt(adagradQuotients[i]);
					if(avgSubgradient[i]<0){
						features[i]=-features[i];
					}
				}
			}			
		}
		else{
			update = learningRate * gradient;
			if(useL2){
				if(sparse){
					long numberOfMissedUpdates = currentTime - timePerFeature[i]; 
					if(numberOfMissedUpdates>0){
						for(int j = 0; j < numberOfMissedUpdates; j++){
							features[i] -= regL2*(features[i] - getRegularizationTarget(i))*learningRate;
						}
					}
				}
				update -= regL2*(features[i] - getRegularizationTarget(i))*learningRate;
			}
		}
		if(useMomentum){
			update += momentum*prevUpdate[i];
			prevUpdate[i] = update;
		}
		if(Double.isNaN(gradient)){
			throw new RuntimeException("nan detected params i=" +i + " gradient=" + gradient);
		}		
		features[i] += update;  
		numberOfUpdates++;

		if(features[i]>maxVal){
			if(sparse){
				long numberOfMissedUpdates = currentTime - timePerFeature[i]; 
				throw new RuntimeException("warning detected weight that is too large: time = " + currentTime + " feature = " + features[i] + " gradient = " + gradient + " update = " + update +  " number of missed updates = " + numberOfMissedUpdates);
			}
			throw new RuntimeException("warning detected weight that is too large: time = " + currentTime + " feature = " + features[i] + " gradient = " + gradient + " update = " + update);
			//			features[i] = maxVal;
		}
		if(features[i]<-maxVal){
			if(sparse){
				long numberOfMissedUpdates = currentTime - timePerFeature[i]; 
				throw new RuntimeException("warning detected weight that is too large: time = " + currentTime + " feature = " + features[i] + " gradient = " + gradient + " update = " + update +  " number of missed updates = " + numberOfMissedUpdates);
			}
			throw new RuntimeException("warning detected weight that is too large: time = " + currentTime + " feature = " + features[i] + " gradient = " + gradient + " update = " + update);
			//			features[i] = -maxVal;
		}
		if(Double.isNaN(features[i])){
			throw new RuntimeException("nan detected params i=" +i + " gradient=" + gradient);
		}
	}

	public void storeGradient(int i, double gradient, int id){
		if(sparse && gradient > 0){
			updatedIndexes.add(i);
		}
		if(!gradientsPerFeature.containsKey(id)){
			gradientsPerFeature.put(id, new double[dim]);
			updatesPerFeature.put(id, new int[dim]);
		}
		gradientsPerFeature.get(id)[i]+=gradient;
		updatesPerFeature.get(id)[i]++;
	}

	public void commitUpdate(double learningRate, double momentum){
		if(!gradientsPerFeature.isEmpty()){
			if(sparse){
				Integer[] keys = gradientsPerFeature.keySet().toArray(new Integer[]{});
				double[] gradientsSum = gradientsPerFeature.get(keys[0]); 
				int[] updatesSum = updatesPerFeature.get(keys[0]);
				for(int k = 1; k < keys.length; k++){
					double[] gradientsForKey = gradientsPerFeature.get(keys[k]);
					int[] updatesForKey = updatesPerFeature.get(keys[k]);
					for(int i : updatedIndexes){
						gradientsSum[i]+=gradientsForKey[i];
						gradientsForKey[i] = 0;
					}
					if(commitMethod == 0){
						for(int i = 0; i < dim; i++){
							updatesSum[i]+=updatesForKey[i];
							updatesForKey[i]=0;
						}
					}
				}
				for(int i : updatedIndexes){
					double gradient = gradientsSum[i];
					if(commitMethod == 0 && updatesSum[i] != 0){
						gradient/=updatesSum[i];
					}
					if(fastUpdate){
						fastUpdate(i, gradient, learningRate);
					}
					else{
						update(i, gradient, learningRate, momentum);
						timePerFeature[i]=currentTime+1;
					}
					gradientsSum[i]=0;
					updatesSum[i]=0;

				}
				currentTime++;
				if(currentTime==Long.MAX_VALUE){
					currentTime = 0;
					timePerFeature = new long[dim];
				}
				updatedIndexes.clear();
			}
			else{
				Integer[] keys = gradientsPerFeature.keySet().toArray(new Integer[]{});
				double[] gradientsSum = gradientsPerFeature.get(keys[0]); 
				int[] updatesSum = updatesPerFeature.get(keys[0]);
				for(int k = 1; k < keys.length; k++){
					double[] gradientsForKey = gradientsPerFeature.get(keys[k]);
					for(int i = 0; i < dim; i++){
						gradientsSum[i]+=gradientsForKey[i];
						gradientsForKey[i] = 0;
					}
					if(commitMethod == 0){
						int[] updatesForKey = updatesPerFeature.get(keys[k]);
						for(int i = 0; i < dim; i++){
							updatesSum[i]+=updatesForKey[i];
							updatesForKey[i]=0;
						}
					}
				}
				for(int i = 0; i < dim; i++){		
					double gradient = gradientsSum[i];
					if(commitMethod == 0 && updatesSum[i] != 0){
						gradient/=updatesSum[i];
					}
					if(fastUpdate){
						fastUpdate(i, gradient, learningRate);
					}
					else{
						update(i, gradient, learningRate, momentum);
					}
					updatesSum[i] = 0;
					gradientsSum[i] = 0;
				}
				currentTime++;
				if(currentTime==Long.MAX_VALUE){
					currentTime = 0;
				}
			}
		}
	}

	public void update(int i, double gradient, double learningRate){
		update(i,gradient, learningRate, 0);
	}

	public double get(int i){
		if(i>=features.length || i < 0){
			throw new RuntimeException("incorrect index " + i + " valid indexes between 0 and " + (features.length-1));
		}
		return features[i];
	}

	public void randInitialize() {
		double max = Math.sqrt(6.0d/dim);
		RandomUtils.initializeRandomArray(features, -max, max, 1);		
	}

	public void randInitialize(int dim) {
		double max = Math.sqrt(6.0d/dim);
		RandomUtils.initializeRandomArray(features, -max, max, 1);		
	}

	//Xavier et Bengio 2010
	public void normalizedInitializationHtan(int fanIn, int fanOut){
		double max = Math.sqrt(6.0d/(fanOut+fanIn));
		RandomUtils.initializeRandomArray(features, -max, max, 1);		
	}

	public void normalizedInitializationSigmoid(int fanIn, int fanOut){
		double max = 4*Math.sqrt(6.0d/(fanOut+fanIn));
		RandomUtils.initializeRandomArray(features, -max, max, 1);		
	}

	public void set(int i, double d) {
		features[i] = d;
	}

	public double[] getArray() {
		return features;
	}

	public void load(double[] loadDoubleArray) {
		features = loadDoubleArray;
		dim = loadDoubleArray.length;
		if(useAdagrad){
			adagradQuotients = new double[dim];
		}
		if(useMomentum){
			prevUpdate = new double[dim];
		}
	}

	public int getDim() {
		return dim;
	}

	public void unlearn(double alpha, int i) {
		double[] base = new double[1];
		RandomUtils.initializeRandomArray(base, -1, 1, Math.sqrt((dim)/6.0d));		
		double diff = features[i] - base[0];

		features[i] -= diff * alpha;		
		if(useAdagrad){
			adagradQuotients[i] -= adagradQuotients[i]*alpha;
		}
	}

	public void unlearn(double alpha) {
		unlearn(alpha, 0, dim-1);
	}

	public void unlearn(double alpha, int start, int end) {
		double[] base = new double[end - start + 1];
		RandomUtils.initializeRandomArray(base, -1, 1, Math.sqrt((dim)/6));		
		for(int i = start; i <= end; i++){
			double diff = features[i] - base[i-start];
			features[i] -= diff * alpha;
			if(useAdagrad){
				adagradQuotients[i] -= adagradQuotients[i]*alpha;
			}
		}
	}

	public void set(FeatureVector vector) {
		this.useAdagrad = vector.useAdagrad;
		this.useMomentum = vector.useMomentum;
		for(int i = 0; i < dim; i++){
			features[i] = vector.get(i);
			if(useAdagrad){
				adagradQuotients[i] = vector.adagradQuotients[i];
			}
			if(useMomentum){
				prevUpdate[i] = vector.prevUpdate[i];
			}
		}		
	}
	
	public void set(double[] vector){
		for(int i = 0; i < dim; i++){
			features[i] = vector[i];
		}
	}

	public void save(PrintStream output){
		SerializeUtils.saveBooleanArray(new boolean[]{useAdagrad, useMomentum, useL1, useL2}, output);
		SerializeUtils.saveDoubleArray(features, output);
		if(useAdagrad){
			SerializeUtils.saveDoubleArray(adagradQuotients, output);
		}
		if(useMomentum){
			SerializeUtils.saveDoubleArray(prevUpdate, output);
		}
	}

	public void load(Scanner reader) {
		boolean[] params = SerializeUtils.loadBooleanArray(reader);
		useAdagrad = params[0];
		useMomentum = params[1];
		useL1 = params[2];
		useL2 = params[3];
		features = SerializeUtils.loadDoubleArray(reader);
		if(useAdagrad){
			adagradQuotients = SerializeUtils.loadDoubleArray(reader);
		}
		if(useMomentum){
			prevUpdate = SerializeUtils.loadDoubleArray(reader);
		}
		dim = features.length;
	}

	public void print() {
		PrintUtils.printDoubleArray("weights", features,false);
		if(useAdagrad){
			PrintUtils.printDoubleArray("adagrad q", adagradQuotients,false);
		}
	}

	public void print(PrintStream out) {
		PrintUtils.printDoubleArray("weights", features,out);
		if(useAdagrad){
			PrintUtils.printDoubleArray("adagrad q", adagradQuotients,out);
		}
	}


	public void reinit() {
		if(useAdagrad){
			for(int i = 0; i < features.length; i++){
				adagradQuotients[i] = 0;
			}
		}

		if(useMomentum){
			for(int i = 0; i < features.length; i++){
				prevUpdate[i] = 0;
			}			
		}
	}


	public void setSparse(boolean sparse) {
		this.sparse = sparse;
		if(sparse){
			timePerFeature = new long[dim];
			currentTime = 0;
		}
	}

	public void setFastUpdate(boolean fastUpdate) {
		FeatureVector.fastUpdate = fastUpdate;
	}

	public void setCommitMethod(int method){
		this.commitMethod = method;
	}
}
