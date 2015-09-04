package jnn.functions.nlp.words;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintStream;

import com.sun.org.apache.xpath.internal.operations.Bool;

import vocab.Vocab;
import jnn.functions.DenseArrayToDenseArrayTransform;
import jnn.functions.composite.DeepRNN;
import jnn.functions.nlp.words.features.CapitalizationWordFeatureExtractor;
import jnn.functions.nlp.words.features.CharSequenceExtractor;
import jnn.functions.nlp.words.features.LowercasedWordFeatureExtractor;
import jnn.functions.nlp.words.features.WordRepresentationSetup;
import jnn.functions.nonparametrized.CopyLayer;
import jnn.functions.nonparametrized.TanSigmoidLayer;
import jnn.functions.parametrized.DenseFullyConnectedLayer;
import jnn.functions.parametrized.Layer;
import jnn.functions.parametrized.StaticLayer;
import jnn.mapping.OutputMappingDenseArrayToDenseArray;
import jnn.mapping.OutputMappingDenseToDense;
import jnn.mapping.OutputMappingStringArrayToDenseArray;
import jnn.mapping.OutputMappingVoidToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.threading.SharedDenseNeuronArray;
import jnn.training.GlobalParameters;
import jnn.training.GraphInference;

public class WordWithContextRepresentation extends Layer implements DenseArrayToDenseArrayTransform{
	private static final String OUTPUT_KEY = "outputkey";
	int inputDim;
	int outputDim;
	
	StaticLayer startOfSentenceInput;
	StaticLayer endOfSentenceInput;
	SharedDenseNeuronArray startOfSentenceNeurons;
	SharedDenseNeuronArray endOfSentenceNeurons;
	GraphInference paddingInference;

	boolean useBLSTM = false; 
	public DeepRNN sequenceRNN;

	boolean useWindow = false;
	public DenseFullyConnectedLayer windowToOutput;
	int windowSize;

	private WordWithContextRepresentation() {
	}
	
	public WordWithContextRepresentation(int inputDim, int outputDim) {
		super();
		this.inputDim = inputDim;
		this.outputDim = outputDim;
		startOfSentenceInput = new StaticLayer(inputDim);
		endOfSentenceInput = new StaticLayer(inputDim);		
		buildPaddingNeurons();
	}

	public void setBLSTMModel(){
		useBLSTM = true;
		useWindow = false;
		sequenceRNN = new DeepRNN(inputDim).addLayer(outputDim, outputDim, "lstm", true, true, 2);
		sequenceRNN.nonlinear = true;
	}
	
	public void setLinearBLSTMModel(){
		useBLSTM = true;
		useWindow = false;
		sequenceRNN = new DeepRNN(inputDim).addLayer(outputDim, outputDim, "lstm", true, true, 2);
		sequenceRNN.nonlinear = false;
	}

	public void setWindowMode(int windowSize){
		useBLSTM = false;
		useWindow = true;
		this.windowSize = windowSize; 
		windowToOutput = new DenseFullyConnectedLayer((windowSize*2+1)*inputDim, outputDim);
	}
	
	private void buildPaddingNeurons(){		
		paddingInference = new GraphInference(0, true);
		DenseNeuronArray startNeurons = new DenseNeuronArray(inputDim);
		DenseNeuronArray endNeurons = new DenseNeuronArray(inputDim);
		paddingInference.addNeurons(1,startNeurons);
		paddingInference.addNeurons(1,endNeurons);
		paddingInference.addMapping(new OutputMappingVoidToDense(startNeurons, startOfSentenceInput));
		paddingInference.addMapping(new OutputMappingVoidToDense(endNeurons, endOfSentenceInput));		
		paddingInference.init();
		paddingInference.forward();
		startOfSentenceNeurons = new SharedDenseNeuronArray(startNeurons);
		endOfSentenceNeurons = new SharedDenseNeuronArray(endNeurons);		
	}
	
	private void updatePaddingNeurons(){
		if(paddingInference != null){
			startOfSentenceNeurons.mergeCopies();
			endOfSentenceNeurons.mergeCopies();
			paddingInference.backward();
		}
	}

	public DenseNeuronArray[] buildOutputs(DenseNeuronArray[] input, GraphInference inference){		
		if(useBLSTM){
			DenseNeuronArray[] inputWithPadding = new DenseNeuronArray[input.length + 2];
			for(int i = 0; i < input.length; i++){
				inputWithPadding[i+1] = input[i];				
				inference.addNeurons(0, input[i]);
			}
			inputWithPadding[0] = startOfSentenceNeurons.getCopy();
			inputWithPadding[inputWithPadding.length-1] = endOfSentenceNeurons.getCopy();
			inference.addNeurons(1,inputWithPadding[0]);
			inference.addNeurons(1,inputWithPadding[inputWithPadding.length-1]);

			DenseNeuronArray[] outputWithPadding = new DenseNeuronArray[inputWithPadding.length];
			for(int i = 0; i < outputWithPadding.length; i++){
				outputWithPadding[i] = new DenseNeuronArray(outputDim);
				inference.addNeurons(2,outputWithPadding[i]);
			}

			inference.addMapping(new OutputMappingDenseArrayToDenseArray(inputWithPadding, outputWithPadding, sequenceRNN));
			inference.init();
			inference.forward();
			DenseNeuronArray[] output = new DenseNeuronArray[input.length];
			for(int i = 0; i < input.length; i++){
				output[i] = outputWithPadding[i+1];
			}
			return output;
		}
		if(useWindow){
			for(int i = 0; i < input.length; i++){
				inference.addNeurons(0, input[i]);
				input[i].setName("input " + i);
			}			
			DenseNeuronArray startPadding = startOfSentenceNeurons.getCopy();
			startPadding.setName("padding start");
			DenseNeuronArray endPadding = endOfSentenceNeurons.getCopy();
			startPadding.setName("padding end");
			inference.addNeurons(1,startPadding);
			inference.addNeurons(1,endPadding);
			DenseNeuronArray[] output = new DenseNeuronArray[input.length];
			for(int i = 0; i < input.length; i++){
				DenseNeuronArray windowInput = new DenseNeuronArray((windowSize*2+1)*inputDim);
				inference.addNeurons(2, windowInput);

				int offset = 0;
				for(int rPos = -windowSize; rPos <= windowSize; rPos++){
					int aPos = i + rPos;
					DenseNeuronArray posNeurons = null;
					if(aPos<0){
						posNeurons = startPadding;
					}
					else if(aPos>=input.length){
						posNeurons = endPadding;
					}
					else{
						posNeurons = input[aPos];
					}
					inference.addMapping(new OutputMappingDenseToDense(0, inputDim-1,offset, offset+inputDim-1,posNeurons, windowInput, CopyLayer.singleton));
					offset+=inputDim;
				}
				DenseNeuronArray windowOutput = new DenseNeuronArray(outputDim);
				inference.addNeurons(3, windowOutput);
				inference.addMapping(new OutputMappingDenseToDense(windowInput, windowOutput, windowToOutput));
				output[i] = new DenseNeuronArray(outputDim);
				inference.addNeurons(4, output[i]);
				inference.addMapping(new OutputMappingDenseToDense(windowOutput, output[i], TanSigmoidLayer.singleton));
			}
			inference.init();
			inference.forward();
			return output;
		}		
		throw new RuntimeException("did nothing in this layer");
	}

	public void buildInference(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray[] output, int outputStart, int outputEnd, OutputMappingDenseArrayToDenseArray mapping){
		DenseNeuronArray[] rep = buildOutputs(input, mapping.getSubInference());
		mapping.setForwardParam(OUTPUT_KEY, rep);
		for(int i = 0; i < input.length; i++){
			output[i].addNeuron(rep[i], 0, outputStart, outputEnd - outputStart + 1);
		}
	}

	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			DenseNeuronArray[] output, int outputStart, int outputEnd,
			OutputMappingDenseArrayToDenseArray mapping) {
		buildInference(input, inputStart, inputEnd, output, outputStart, outputEnd, mapping);
	}

	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, DenseNeuronArray[] output, int outputStart,
			int outputEnd, OutputMappingDenseArrayToDenseArray mapping) {
		DenseNeuronArray[] rep = (DenseNeuronArray[]) mapping.getForwardParam(OUTPUT_KEY);
		for(int i = 0; i < input.length; i++){
			rep[i].addError(output[i], outputStart, 0, outputEnd - outputStart + 1);
		}
		mapping.getSubInference().backward();
	}
	
	@Override
	public void updateWeights(double learningRate, double momentum) {
		updatePaddingNeurons();
		buildPaddingNeurons();
		startOfSentenceInput.updateWeights(learningRate, momentum);
		endOfSentenceInput.updateWeights(learningRate, momentum);
		if(useBLSTM){
			sequenceRNN.updateWeights(learningRate, momentum);
		}
		if(useWindow){
			windowToOutput.updateWeights(learningRate, momentum);
		}
	}
	
	public int getOutputDim() {
		return outputDim;
	}
	
	public void save(PrintStream out){
		out.println(inputDim);
		out.println(outputDim);
		startOfSentenceInput.save(out);
		endOfSentenceInput.save(out);
		
		out.println(useBLSTM);
		out.println(useWindow);
		out.println(windowSize);
		if(useBLSTM){
			sequenceRNN.save(out);
		}
		else if(useWindow){
			windowToOutput.save(out);
		}
		else{
			throw new RuntimeException("unknown context model");
		}
	}
	
	public static WordWithContextRepresentation load(BufferedReader in){
		try {
			WordWithContextRepresentation rep = new WordWithContextRepresentation();
			rep.inputDim = Integer.parseInt(in.readLine());
			rep.outputDim = Integer.parseInt(in.readLine());
			rep.startOfSentenceInput = StaticLayer.load(in);
			rep.endOfSentenceInput = StaticLayer.load(in);
			rep.useBLSTM = Boolean.parseBoolean(in.readLine());
			rep.useWindow = Boolean.parseBoolean(in.readLine());
			rep.windowSize = Integer.parseInt(in.readLine());
			rep.buildPaddingNeurons();
			if(rep.useBLSTM){
				rep.sequenceRNN = DeepRNN.load(in);
			}
			else if(rep.useWindow){
				rep.windowToOutput = DenseFullyConnectedLayer.load(in);
			}
			else{
				throw new RuntimeException("unknown context model");
			}
			
			return rep;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public static void main(String[] args){
		GlobalParameters.useMomentumDefault = true;
		Vocab vocab = new Vocab();
		vocab.addWordToVocab("I");
		vocab.addWordToVocab("went");
		vocab.addWordToVocab("shopping");
		vocab.addWordToVocab("in");
		vocab.addWordToVocab("Lisbonnnnn");

		WordRepresentationSetup setup = new WordRepresentationSetup(vocab, 50, 50, 150);
		setup.addFeatureExtractor(new LowercasedWordFeatureExtractor(), null);
		setup.addFeatureExtractor(new CapitalizationWordFeatureExtractor());
		setup.addSequenceExtractor(new CharSequenceExtractor());
		WordRepresentationLayer layer = new WordRepresentationLayer(setup);

		WordWithContextRepresentation context = new WordWithContextRepresentation(layer.getOutputDim(), 50);
		context.setBLSTMModel();
		//context.setWindowMode(5);
		for(int j = 0; j < 10; j++){
		for(int e = 0; e < 100; e++){
			GraphInference inference = new GraphInference(0, true);
			String[] input = "I am living in Lisbonnnnn".split("\\s+");
			DenseNeuronArray[] representation = DenseNeuronArray.asArray(input.length, layer.getOutputDim());
			inference.addNeurons(1, representation);
			inference.addMapping(new OutputMappingStringArrayToDenseArray(input, representation, layer));

			DenseNeuronArray[] contextRepresentation = DenseNeuronArray.asArray(input.length, 50, "context rep");
			inference.addNeurons(2, contextRepresentation);
			inference.addMapping(new OutputMappingDenseArrayToDenseArray(representation, contextRepresentation, context));

			inference.init();
			inference.forward();
			for(int w = 0; w < input.length; w++){
				for(int i = 0; i < 50; i++){
					contextRepresentation[w].addError(i, 1-contextRepresentation[w].getNeuron(i));
				}
			}
			inference.backward();
			//inference.printNeurons();
		}
		context.updateWeights(0, 0);
		}
	}
}
