package jnn.functions.nlp.app.lm;

import java.util.ArrayList;

import jnn.functions.composite.LookupTable;
import jnn.functions.composite.lstm.LSTMDecoder;
import jnn.functions.composite.lstm.LSTMDecoderState;
import jnn.functions.composite.lstm.aux.LSTMMapping;
import jnn.functions.nlp.aux.input.InputSentence;
import jnn.functions.nlp.words.OutputWordRepresentationLayer;
import jnn.functions.nlp.words.WordRepresentationLayer;
import jnn.functions.nonparametrized.CopyLayer;
import jnn.functions.nonparametrized.SoftmaxLayer;
import jnn.functions.nonparametrized.WeightedSumLayer;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.StaticLayer;
import jnn.mapping.OutputMappingDenseArrayToDense;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.mapping.OutputMappingDenseArrayToStringArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingStringArrayToDenseArray;
import jnn.mapping.OutputMappingStringToDense;
import jnn.mapping.OutputMappingVoidToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.StringNeuronArray;
import jnn.objective.WordSoftmaxSparseObjective;
import jnn.training.GraphInference;

public class AttentionBowLanguageModelInstance {

	public int id;
	public WordRepresentationLayer wordRepresentation;
	public WordRepresentationLayer wordAttention;
	public StaticLayer positionBias;
	public OutputWordRepresentationLayer outputWordRepresentation;
	double norm;	
	String SOS;
	String EOS;
	int windowSize;
	int bowSize;
	public DenseFullyConnectedLayer wordToAttention;

	InputSentence[] batchSents;

	long inferenceTime = 0;
	long buildUpTime = 0;
	long forwardTime = 0;
	long backwardTime = 0;
	ArrayList<WordSoftmaxSparseObjective> objectives;
	
	double loglikelihood = 0;
	double loglikelihoodUnk = 0;
	double loglikelihoodKnown = 0;
	int words = 0;
	int wordUnk = 0;
	int wordKnown = 0;
	double[] loglikelihoodPerSentence;
	int mode;
	
	public AttentionBowLanguageModelInstance(int id, double norm, InputSentence[] batchSents, WordRepresentationLayer wordRep,
			WordRepresentationLayer wordAttention, DenseFullyConnectedLayer wordToAttention, OutputWordRepresentationLayer softmax,StaticLayer positionBias, String SOS, String EOS, int windowSize, int bowSize, int mode) {
		super();
		this.id = id;
		this.batchSents = batchSents;
		this.wordRepresentation = wordRep;
		this.norm = norm;
		this.wordAttention = wordAttention;
		this.wordToAttention = wordToAttention;
		this.outputWordRepresentation = softmax;
		this.SOS = SOS;
		this.EOS = EOS;
		this.positionBias = positionBias;
		this.loglikelihoodPerSentence = new double[batchSents.length];
		this.bowSize = bowSize;
		this.windowSize = windowSize;
		this.mode = mode;
	}
	
	public void buildNetwork(int index, InputSentence sent, GraphInference inference){
		int inputSize = wordRepresentation.getOutputDim();
		long buildStart = System.currentTimeMillis();
		String[] tokens = sent.tokens;
		String[] tokensWithSOS = new String[tokens.length+1];
		String[] tokensWithEOS = new String[tokens.length+1];
		for(int i = 0; i < tokens.length; i++){
			tokensWithSOS[i+1] = tokens[i];
			tokensWithEOS[i] = tokens[i];
		}
		tokensWithSOS[0] = SOS; 
		tokensWithEOS[tokens.length] = EOS;
		DenseNeuronArray[] inputReps = DenseNeuronArray.asArray(tokensWithSOS.length, wordRepresentation.getOutputDim());
		inference.addNeurons(inputReps);
		inference.addMapping(new OutputMappingStringArrayToDenseArray(tokensWithSOS, inputReps, wordRepresentation));

		DenseNeuronArray[] inputAttentions = DenseNeuronArray.asArray(tokensWithSOS.length, bowSize - windowSize + 1);
		inference.addNeurons(inputAttentions);		
		if(mode == 0){
			inference.addMapping(new OutputMappingStringArrayToDenseArray(tokensWithSOS, inputAttentions, wordAttention));
		}
		else{
			inference.addMapping(new OutputMappingDenseArrayToDenseArray(inputReps, inputAttentions, wordToAttention));			
		}
		
		DenseNeuronArray[] bowscores = DenseNeuronArray.asArray(tokensWithSOS.length, positionBias.outputDim, "scores");
		inference.addNeurons(bowscores);
		
		DenseNeuronArray[] bowsoftmax = DenseNeuronArray.asArray(tokensWithSOS.length, positionBias.outputDim, "softmax");
		inference.addNeurons(bowsoftmax);

		DenseNeuronArray[] contextReps = DenseNeuronArray.asArray(tokensWithSOS.length, outputWordRepresentation.getInputDim());
		inference.addNeurons(contextReps);
		
		DenseNeuronArray emptyInput = new DenseNeuronArray(inputSize);
		inference.addNeurons(emptyInput);
		emptyInput.init();
		
		for(int i = 0; i < tokensWithSOS.length; i++){
			DenseNeuronArray context = contextReps[i];
			//adding window words
			DenseNeuronArray[] bowInputArray = new DenseNeuronArray[bowSize - windowSize + 1];
			for(int w = 0; w < windowSize; w++){
				if(i-w < 0){
					
				}
				else{
					inference.addMapping(new OutputMappingDenseToDense(0,inputSize-1,w*inputSize, (w+1)*inputSize-1, inputReps[i-w], context, CopyLayer.singleton));
				}
			}
						
			inference.addMapping(new OutputMappingVoidToDense(bowscores[i], positionBias));
			for(int b = windowSize; b < bowSize; b++){
				if(i-b < 0){
					bowInputArray[b-windowSize] = emptyInput;
				}
				else{
					inference.addMapping(new OutputMappingDenseToDense(b-windowSize,b-windowSize, b-windowSize,b-windowSize,inputAttentions[i],bowscores[i],CopyLayer.singleton));
					bowInputArray[b-windowSize] = inputReps[i-b];					
				}
			}
			
			inference.addMapping(new OutputMappingDenseToDense(bowscores[i], bowsoftmax[i], SoftmaxLayer.singleton));
			bowInputArray[bowSize - windowSize] = bowsoftmax[i];
			
			inference.addMapping(new OutputMappingDenseArrayToDense(0,inputSize-1,windowSize * inputSize, (windowSize+1)*inputSize-1,bowInputArray, context, WeightedSumLayer.singleton));
		}
		
		StringNeuronArray[] outputWords = StringNeuronArray.asArray(tokensWithEOS.length);
		inference.addNeurons(outputWords);
		inference.addMapping(new OutputMappingDenseArrayToStringArray(contextReps, outputWords, outputWordRepresentation));
		
		StringNeuronArray.setExpectedArray(tokensWithEOS, outputWords);
		inference.init();
		long buildEnd = System.currentTimeMillis();
		buildUpTime+=buildEnd-buildStart;
		forwardTime+=inference.forward();

		double sentLL = 0;
		for(int i = 0; i < outputWords.length; i++){
			sentLL += outputWords[i].getScore();
			words++;
			if(outputWordRepresentation.isOOV(outputWords[i].expected)){
				loglikelihoodUnk+=outputWords[i].getScore();
				wordUnk++;
			}
			else{
				loglikelihoodKnown+=outputWords[i].getScore();
				wordKnown++;
			}
		}
		loglikelihood += sentLL;
		loglikelihoodPerSentence[index] = sentLL;
	}
	
	public void train(){
		int i = 0;
		for(InputSentence sent : batchSents){
			GraphInference inference = new GraphInference(id, true);
			inference.setNorm(norm);
			buildNetwork(i++, sent, inference);
			backwardTime+=inference.backward();
			inferenceTime += inference.getInferenceTime();
		}
	}	
	
	public void computeLL(){
		int i = 0;
		for(InputSentence sent : batchSents){
			GraphInference inference = new GraphInference(id, false);
			inference.setNorm(norm);
			buildNetwork(i++, sent, inference);
		}		
	}
}
