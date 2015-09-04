package jnn.functions.nlp.app.lm;

import java.util.ArrayList;

import jnn.functions.composite.lstm.LSTMDecoder;
import jnn.functions.composite.lstm.LSTMDecoderState;
import jnn.functions.composite.lstm.aux.LSTMMapping;
import jnn.functions.nlp.aux.input.InputSentence;
import jnn.functions.nlp.words.OutputWordRepresentationLayer;
import jnn.functions.nlp.words.WordRepresentationLayer;
import jnn.functions.parametrized.StaticLayer;
import jnn.mapping.OutputMappingDenseArrayToStringArray;
import jnn.mapping.OutputMappingStringArrayToDenseArray;
import jnn.mapping.OutputMappingVoidToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.StringNeuronArray;
import jnn.objective.WordSoftmaxSparseObjective;
import jnn.training.GraphInference;

public class LSTMLanguageModelInstance {

	public int id;
	public WordRepresentationLayer wordRepresentation;
	public LSTMDecoder decoder;
	public OutputWordRepresentationLayer outputWordRepresentation;
	public StaticLayer initialStateLayer;
	public StaticLayer initialCellLayer;
	double norm;	
	String SOS;
	String EOS;

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

	public LSTMLanguageModelInstance(int id, double norm, InputSentence[] batchSents, WordRepresentationLayer wordRep,
			LSTMDecoder decoder, OutputWordRepresentationLayer softmax,StaticLayer initialState, StaticLayer initialCell, String SOS, String EOS) {
		super();
		this.id = id;
		this.batchSents = batchSents;
		this.wordRepresentation = wordRep;
		this.norm = norm;
		this.decoder = decoder;
		this.outputWordRepresentation = softmax;
		this.SOS = SOS;
		this.EOS = EOS;
		this.initialCellLayer = initialCell;
		this.initialStateLayer = initialState;
		this.loglikelihoodPerSentence = new double[batchSents.length];
	}

	public void buildNetwork(int index, InputSentence sent, GraphInference inference){
		long buildStart = System.currentTimeMillis();
		String[] tokens = sent.tokens;
		String[] tokensWithSOS = new String[tokens.length+1];
		String[] tokensWithEOS = new String[tokens.length+1];
		for(int i = 0; i < tokens.length; i++){
			tokensWithSOS[i+1] = tokens[i];
			tokensWithEOS[i] = tokens[i].toLowerCase();
		}
		tokensWithSOS[0] = SOS; 
		tokensWithEOS[tokens.length] = EOS;
		DenseNeuronArray[] inputReps = DenseNeuronArray.asArray(tokensWithSOS.length, wordRepresentation.getOutputDim());
		inference.addNeurons(inputReps);
		inference.addMapping(new OutputMappingStringArrayToDenseArray(tokensWithSOS, inputReps, wordRepresentation));
		
		DenseNeuronArray initialState = new DenseNeuronArray(decoder.getStateDim());
		DenseNeuronArray initialCell = new DenseNeuronArray(decoder.getStateDim());
		inference.addNeurons(initialCell);
		inference.addNeurons(initialState);
		inference.addMapping(new OutputMappingVoidToDense(initialCell, initialCellLayer));
		inference.addMapping(new OutputMappingVoidToDense(initialState, initialStateLayer));
		
		LSTMDecoderState[] states = LSTMDecoderState.buildStateSequence(inputReps, decoder.getStateDim());
		inference.addNeurons(LSTMDecoderState.getStates(states));
		inference.addNeurons(LSTMDecoderState.getCells(states));

		LSTMMapping lstmMapping = new LSTMMapping(states, initialState, initialCell, decoder);
		inference.addMapping(lstmMapping);
		
		StringNeuronArray[] outputWords = StringNeuronArray.asArray(tokensWithEOS.length);
		inference.addNeurons(outputWords);
		inference.addMapping(new OutputMappingDenseArrayToStringArray(LSTMDecoderState.getStates(states), outputWords, outputWordRepresentation));
		
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
