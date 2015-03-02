package jnn.objective;

import jnn.neuron.DenseNeuronArray;
import util.MathUtils;
import util.PrintUtils;

public class SentenceSoftmaxDenseObjective {
	public static boolean SLL_DEFAULT = false;

	DenseNeuronArray[] nodeActivationsPerTimestamp; 
	DenseNeuronArray[] transitionActivations; // time invariant
	int[] expectedIndexesPerTimestamp;
	double[][] probsPerTimestamp;
	int numberOfTags;
	int length;

	double[][][] nodeTransitionScores; //exp(node1 + transition12)

	int[] predictions;
	double[][] viterbiScores = new double[length][numberOfTags];
	int[][] viterbiChoices = new int[length][numberOfTags];

	boolean useSentenceLikelihood = SLL_DEFAULT;

	public SentenceSoftmaxDenseObjective(
			DenseNeuronArray[] nodeActivationsPerTimestamp,
			DenseNeuronArray[] transitionActivations) {
		super();
		this.nodeActivationsPerTimestamp = nodeActivationsPerTimestamp;
		this.transitionActivations = transitionActivations;
		numberOfTags=nodeActivationsPerTimestamp[0].size;
		length=nodeActivationsPerTimestamp.length;		
	}

	public void setExpected(int[] expected){
		this.expectedIndexesPerTimestamp = expected;
		probsPerTimestamp = new double[length][numberOfTags];
		for(int i = 0; i < length; i++){
			probsPerTimestamp[i][expectedIndexesPerTimestamp[i]] = 1.0d;
		}
	}

	public void setExpected(double[][] probs){
		this.probsPerTimestamp = probs;
		expectedIndexesPerTimestamp = new int[length];
		for(int i = 0; i < length; i++){
			expectedIndexesPerTimestamp[i] = MathUtils.maxIndex(probsPerTimestamp[i]);
		}
	}

	public double[][] computeProbPerNode(){
		calcPotentials();
		double[][] betas = new double[length][numberOfTags];	

		double[] endTransitions = new double[numberOfTags];
		for(int i = 0; i < numberOfTags; i++){
			endTransitions[i] = transitionActivations[i].getNeuron(numberOfTags);
		}
		betas[length-1] = getProbabilities(nodeActivationsPerTimestamp[length-1],endTransitions);

		for(int t = length - 1; t > 0; t--){
			for(int i = 0; i < numberOfTags; i++){
				betas[t-1][i] = 0;				
				for(int j = 0; j < numberOfTags; j++){					
					betas[t-1][i]+=betas[t][j]*nodeTransitionScores[t-1][i][j];
				}
				if(betas[t-1][i] == 0){
					throw new RuntimeException("activation equal to zero sentence length =" + length);
				}
			}
		}
		return betas;
	}

	public double addError() {
		return addError(1.0);
	}

	public double addError(double norm) {
		if(useSentenceLikelihood){
			calcPotentials();
			int[][] sumNode = new int[length][numberOfTags];
			int[][] sumTransition = new int[numberOfTags+1][numberOfTags+1];

			double[][] sumLogaddTransition = new double[numberOfTags+1][numberOfTags+1];

			for(int t = length-1; t>=0; t--){
				int expected = expectedIndexesPerTimestamp[t];
				int prevExpected = numberOfTags; 
				if(t>0){
					prevExpected = expectedIndexesPerTimestamp[t-1];				
				}
				sumNode[t][expected]++;
				sumTransition[prevExpected][expected]++;			
			}
			sumTransition[expectedIndexesPerTimestamp[length-1]][numberOfTags]++;

			double[][] betas = new double[length][numberOfTags];	

			double[] endTransitions = new double[numberOfTags];
			for(int i = 0; i < numberOfTags; i++){
				endTransitions[i] = transitionActivations[i].getNeuron(numberOfTags);
			}
			betas[length-1] = getProbabilities(nodeActivationsPerTimestamp[length-1],endTransitions);

			for(int t = length - 1; t > 0; t--){
				for(int i = 0; i < numberOfTags; i++){
					betas[t-1][i] = 0;				
					for(int j = 0; j < numberOfTags; j++){					

						betas[t-1][i]+=betas[t][j]*nodeTransitionScores[t-1][i][j];
						//transitions
						sumLogaddTransition[i][j]+=betas[t][j]*nodeTransitionScores[t-1][i][j];
					}
				}
			}

			for(int i = 0; i < numberOfTags; i++){
				sumLogaddTransition[i][numberOfTags] += betas[length-1][i];
			}

			for(int i = 0; i < numberOfTags; i++){
				sumLogaddTransition[numberOfTags][i] += betas[0][i];
			}

			double error = 0; 
			//nodes
			for(int t = length - 1; t >= 0; t--){
				for(int o = 0; o < numberOfTags; o++){
					double nodeError = (sumNode[t][o] - betas[t][o])*norm;
					nodeActivationsPerTimestamp[t].addError(o, nodeError);
					error+=Math.abs(nodeError);
				}
			}

			//transitions
			for(int i = 0; i < numberOfTags+1; i++){
				for(int j = 0; j < numberOfTags+1; j++){
					transitionActivations[i].addError(j, (sumTransition[i][j] - sumLogaddTransition[i][j])*norm);
				}
			}		

			//initial transitions
			//		DenseNeuronArray nodeProbs = nodeActivationsPerTimestamp[0];
			//		double[] expPerObjective = new double[numberOfTags];
			//		for(int j = 0; j < numberOfTags; j++){
			//			expPerObjective[j] = Math.exp(nodeProbs.getNeuron(j));
			//		}
			//		double norm = MathUtils.sum(expPerObjective);
			//	
			return error;
		}
		else{
			double error = 0;
			double[][] exp = new double[length][numberOfTags];
			for(int i = 0; i < length; i++){
				double[] expI = exp[i];
				double max = nodeActivationsPerTimestamp[i].getNeuron(expectedIndexesPerTimestamp[i]);				
//				double max = 0;				
				for(int tag = 0; tag < numberOfTags; tag++){
					expI[tag] = Math.exp(nodeActivationsPerTimestamp[i].getNeuron(tag) - max);					
				}
				double sum = MathUtils.sum(expI);
				for(int tag = 0; tag < numberOfTags; tag++){
					nodeActivationsPerTimestamp[i].addError(tag, norm*(-expI[tag]/sum));
				}
				nodeActivationsPerTimestamp[i].addError(expectedIndexesPerTimestamp[i], norm);
			}
			return error;
		}
	}

	public double[] getProbabilities(DenseNeuronArray nodeActivations){		
		double[] expArray = new double[numberOfTags];
		for(int i = 0; i < numberOfTags; i++){
			expArray[i] = Math.exp(nodeActivations.getNeuron(i));
		}
		return MathUtils.normVectorTo1(expArray);		
	}

	public double[] getProbabilities(DenseNeuronArray nodeActivations, double[] transitionActivations){
		double[] expArray = new double[numberOfTags];
		for(int i = 0; i < numberOfTags; i++){
			expArray[i] = Math.exp(nodeActivations.getNeuron(i) + transitionActivations[i]);
		}
		return MathUtils.normVectorTo1(expArray);		
	}

	public int getClassError(int index) {
		if(useSentenceLikelihood){
			viterbi();
			if(predictions[index] != expectedIndexesPerTimestamp[index]){
				return 1;
			}
			return 0;
		}
		else{			
			int max = nodeActivationsPerTimestamp[index].maxIndex();
			if(max != expectedIndexesPerTimestamp[index]){
				return 1;
			}
			return 0;
		}
	}

	public int getLength() {
		return length;
	}

	public int getPrediction(int w) {
		if(useSentenceLikelihood){
			viterbi();
			return predictions[w];
		}
		else{
			int max = nodeActivationsPerTimestamp[w].maxIndex();
			return max;
		}
	}

	public void calcPotentials() {
		if(nodeTransitionScores == null){
			nodeTransitionScores = new double[length][numberOfTags][numberOfTags];
			for(int t = 0; t < length; t++){
				DenseNeuronArray nodeProbs = nodeActivationsPerTimestamp[t];
				for(int j = 0; j < numberOfTags; j++){
					double[] expPerObjective = new double[numberOfTags];
					for(int k = 0; k < numberOfTags; k++){
						double transitionActivation = 0;
						if(t!=length-1){
							transitionActivation = transitionActivations[k].getNeuron(j);
						}
						if(t==1){
							expPerObjective[k] = Math.exp(nodeProbs.getNeuron(k) + transitionActivation + transitionActivations[numberOfTags].getNeuron(k));
						}
						else{	
							expPerObjective[k] = Math.exp(nodeProbs.getNeuron(k) + transitionActivation);
						}
					}
					double norm = MathUtils.sum(expPerObjective);
					for(int k = 0; k < numberOfTags; k++){
						nodeTransitionScores[t][k][j] = expPerObjective[k]/norm;						
					}
				}
			}
		}
	}

	public void viterbi(){
		if(predictions == null){
			calcPotentials();
			predictions = new int[length];
			viterbiScores = new double[length][numberOfTags];
			viterbiChoices = new int[length][numberOfTags];

			double[] endTransitions = new double[numberOfTags];
			for(int i = 0; i < numberOfTags; i++){
				endTransitions[i] = transitionActivations[i].getNeuron(numberOfTags);
			}
			viterbiScores[length-1] = getProbabilities(nodeActivationsPerTimestamp[length-1],endTransitions);

			for(int t = length - 1; t > 0; t--){
				for(int i = 0; i < numberOfTags; i++){
					viterbiScores[t-1][i] = -Double.MAX_VALUE;
					viterbiChoices[t-1][i] = -1;
					for(int j = 0; j < numberOfTags; j++){
						double scoreItoJ = viterbiScores[t][j]*nodeTransitionScores[t-1][i][j];						
						if(scoreItoJ>viterbiScores[t-1][i]){
							viterbiScores[t-1][i] = scoreItoJ;
							viterbiChoices[t-1][i] = j; 
						}
						if(scoreItoJ <= 0){
							throw new RuntimeException("path returned a negative score, sentence size = " + length);
						}
					}
				}
			}
			predictions[0] = MathUtils.maxIndex(viterbiScores[0]);
			for(int w = 0; w < length-1;w++){
				predictions[w+1] = viterbiChoices[w][predictions[w]];
			}
		}
	}

	public void printStats(){
		viterbi();
		String seq = "";
		for(int i = 0 ; i < predictions.length; i++){
			seq+=predictions[i] + " ";
		}
		System.err.println("viterbi sequence " + seq);
		System.err.println();
		PrintUtils.printDoubleMatrix("viterbi scores", viterbiScores, false);

		System.err.println();
		for(int i = 0 ; i < predictions.length; i++){
			PrintUtils.printDoubleMatrix("potential scores at w " + i, nodeTransitionScores[i], false);			
		}

	}

	public static void main(String[] args){
		int length = 100;
		int numTags = 45;
		DenseNeuronArray[] input = new DenseNeuronArray[length];		

		System.err.println("input");
		for(int i = 0; i < input.length; i++){
			input[i] = new DenseNeuronArray(numTags);
			input[i].init();
			for(int j = 0; j < numTags; j++){
				input[i].addNeuron(j, j*(i+1)/(double)numTags);
			}
		}

		System.err.println("transition");
		DenseNeuronArray[] transition = new DenseNeuronArray[numTags+1];
		for(int i = 0; i < numTags+1; i++){
			transition[i] = new DenseNeuronArray(numTags+1);
			transition[i].init();
			System.err.println(transition[i]);
		}		

		SentenceSoftmaxDenseObjective obj = new SentenceSoftmaxDenseObjective(input, transition);
		int[] expected = new int[length];
		for(int i = 0 ; i < expected.length; i++){
			expected[i] = 1;
		}
		obj.setExpected(expected);
//		obj.printStats();
		obj.addError(5);

		System.err.println("input");
		for(int i = 0; i < input.length; i++){
			System.err.println(input[i]);
		}

		System.err.println("transition");
		for(int i = 0; i < numTags+1; i++){
			System.err.println(transition[i]);
		}		

		PrintUtils.printDoubleMatrix("probs ", obj.computeProbPerNode(), false);
		for(int i = 0; i < input.length; i++){
			System.err.println(input[i]);
		}
		for(int i = 0; i < input.length; i++){
			System.err.println(obj.getClassError(i));
		}
	}	
}
