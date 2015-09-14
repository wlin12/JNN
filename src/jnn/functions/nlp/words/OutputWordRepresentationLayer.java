package jnn.functions.nlp.words;

import java.io.BufferedReader;
import java.io.PrintStream;
import java.util.HashSet;

import util.TopNList;
import vocab.Vocab;
import vocab.WordEntry;
import jnn.functions.DenseArrayToStringArrayTransform;
import jnn.functions.DenseToStringTransform;
import jnn.functions.StringArrayToDenseArrayTransform;
import jnn.functions.StringToDenseTransform;
import jnn.functions.composite.AbstractSofmaxObjectiveLayer;
import jnn.functions.composite.HierarchicalSoftmaxObjectiveLayer;
import jnn.functions.composite.NoiseConstrastiveEstimationLayer;
import jnn.functions.composite.SoftmaxObjectiveLayer;
import jnn.functions.nlp.words.features.CharSequenceExtractor;
import jnn.functions.nlp.words.features.OutputWordRepresentationSetup;
import jnn.functions.nlp.words.features.WordRepresentationSetup;
import jnn.functions.parametrized.Layer;
import jnn.mapping.OutputMappingDenseArrayToStringArray;
import jnn.mapping.OutputMappingDenseToString;
import jnn.mapping.OutputMappingStringArrayToDenseArray;
import jnn.mapping.OutputMappingStringToDense;
import jnn.neuron.DenseNeuronArray;
import jnn.neuron.StringNeuronArray;
import jnn.training.GraphInference;

public class OutputWordRepresentationLayer extends Layer implements DenseToStringTransform, DenseArrayToStringArrayTransform{
	
	AbstractSofmaxObjectiveLayer wordSoftmax;	
	OutputWordRepresentationSetup setup;	
	
	public OutputWordRepresentationLayer(OutputWordRepresentationSetup setup) {
		super();
		this.setup = setup;
		if(setup.sequenceType.startsWith("word")){
			if(setup.sequenceType.equals("word-nce")){
				wordSoftmax = new NoiseConstrastiveEstimationLayer(setup.inputDim, setup.existingWords, setup.negativeSamplingRate, setup.UNK);
			}
			else if(setup.sequenceType.equals("word-leave-1-out")){
				Vocab vocab = new Vocab();
				for(int i = 0; i < setup.existingWords.getTypes(); i++){
					WordEntry entry = setup.existingWords.getEntryFromId(i);
					if(entry.getCount()>1){
						vocab.addWordToVocab(entry.getWord(), entry.getCount());
					}
					else{
						vocab.addWordToVocab(setup.UNK, entry.getCount());
					}
				}
				vocab.sortVocabByCount();
				wordSoftmax = new SoftmaxObjectiveLayer(vocab, setup.inputDim,  setup.UNK);
			}
			else if(setup.sequenceType.equals("word-nce-leave-1-out")){
				Vocab vocab = new Vocab();
				for(int i = 0; i < setup.existingWords.getTypes(); i++){
					WordEntry entry = setup.existingWords.getEntryFromId(i);
					if(entry.getCount()>1){
						vocab.addWordToVocab(entry.getWord(), entry.getCount());
					}
					else{
						vocab.addWordToVocab(setup.UNK, entry.getCount());
					}
				}
				vocab.sortVocabByCount();
				wordSoftmax = new NoiseConstrastiveEstimationLayer(setup.inputDim, vocab, setup.negativeSamplingRate, setup.UNK);
			}
			else if(setup.sequenceType.equals("word")){
				wordSoftmax = new SoftmaxObjectiveLayer(setup.existingWords, setup.inputDim,  setup.UNK);
			}
			else if(setup.sequenceType.equals("word-hier")){
				wordSoftmax = new HierarchicalSoftmaxObjectiveLayer(setup.existingWords, setup.inputDim, setup.UNK);
			}
			else {
				int limit = Integer.parseInt(setup.sequenceType.split("-")[1]);
				Vocab vocab = new Vocab();
				for(int i = 0; i < setup.existingWords.getTypes(); i++){
					WordEntry entry = setup.existingWords.getEntryFromId(i);
					vocab.addWordToVocab(entry.getWord(), entry.getCount());
				}
				HashSet<String> exceptions = new HashSet<String>();
				exceptions.add(setup.UNK);
				vocab.sortVocabByCount(limit,exceptions);
				wordSoftmax = new SoftmaxObjectiveLayer(vocab, setup.inputDim,  setup.UNK);
			}
		}		
		else if(setup.sequenceType.equals("character")){
			WordFromCharacterSoftmax wordFromCharacterSoftmax = new WordFromCharacterSoftmax(setup.existingWords,0, setup.inputDim, setup.characterDim, setup.characterDim,0);
			wordFromCharacterSoftmax.beamSize = setup.beam;
			wordSoftmax = wordFromCharacterSoftmax;
		}
		else if(setup.sequenceType.equals("character-hier")){
			WordFromCharacterSoftmax wordFromCharacterSoftmax = new WordFromCharacterSoftmax(setup.existingWords,1, setup.inputDim, setup.characterDim, setup.characterDim,0);
			wordFromCharacterSoftmax.beamSize = setup.beam;
			wordSoftmax = wordFromCharacterSoftmax;
		}
		else if(setup.sequenceType.equals("character-nce")){
			WordFromCharacterSoftmax wordFromCharacterSoftmax = new WordFromCharacterSoftmax(setup.existingWords,2, setup.inputDim, setup.characterDim, setup.characterDim,setup.negativeSamplingRate);
			wordFromCharacterSoftmax.beamSize = setup.beam;
			wordSoftmax = wordFromCharacterSoftmax;
		}
		else{
			throw new RuntimeException("unknown softmax type: " + setup.sequenceType);
		}
	}
	
	public boolean isOOV(String word){
		return setup.existingWords.getEntry(word) == null;
	}

	@Override
	public void updateWeights(double learningRate, double momentum) {
		wordSoftmax.updateWeights(learningRate, momentum);
	}
	
	@Override
	public void forward(DenseNeuronArray input, int inputStart, int inputEnd,
			StringNeuronArray output, OutputMappingDenseToString mapping) {
		wordSoftmax.forward(input, inputStart, inputEnd, output, mapping);	
	}
	
	@Override
	public void forward(DenseNeuronArray[] input, int inputStart, int inputEnd,
			StringNeuronArray[] output,
			OutputMappingDenseArrayToStringArray mapping) {
		wordSoftmax.forward(input, inputStart, inputEnd, output, mapping);
	}
	
	@Override
	public void backward(DenseNeuronArray input, int inputStart, int inputEnd,
			StringNeuronArray output, OutputMappingDenseToString mapping) {
		wordSoftmax.backward(input, inputStart, inputEnd, output, mapping);
	}
	
	@Override
	public void backward(DenseNeuronArray[] input, int inputStart,
			int inputEnd, StringNeuronArray[] output,
			OutputMappingDenseArrayToStringArray mapping) {
		wordSoftmax.backward(input, inputStart, inputEnd, output, mapping);
	}
	
	public String decode(DenseNeuronArray input){
		return wordSoftmax.decode(input);
	}

	public TopNList<String> getTopN(DenseNeuronArray input, int n){
		return wordSoftmax.getTopN(input, n);
	}
	
	public void save(PrintStream out){
		wordSoftmax.save(out);
	}
	
	public static OutputWordRepresentationLayer load(BufferedReader in, OutputWordRepresentationSetup setup){
		OutputWordRepresentationLayer softmax = new OutputWordRepresentationLayer(setup);
		if(setup.sequenceType.startsWith("word")){
			if(setup.sequenceType.equals("word-nce")){
				softmax.wordSoftmax = NoiseConstrastiveEstimationLayer.load(in);
			}
			else if(setup.sequenceType.equals("word")){
				softmax.wordSoftmax = SoftmaxObjectiveLayer.load(in);
			}
			else if(setup.sequenceType.equals("word-hier")){
				softmax.wordSoftmax = HierarchicalSoftmaxObjectiveLayer.load(in);
			}
			else {
				softmax.wordSoftmax = SoftmaxObjectiveLayer.load(in);
			}
		}			
		else if(setup.sequenceType.equals("character") || setup.sequenceType.equals("character-hier") || setup.sequenceType.equals("character-nce")){
			WordFromCharacterSoftmax characterSM = WordFromCharacterSoftmax.load(in);
			characterSM.beamSize = setup.beam;
			softmax.wordSoftmax = characterSM;
		}
		
		else if(setup.sequenceType.equals("word-nce")){
			softmax.wordSoftmax = NoiseConstrastiveEstimationLayer.load(in);
		}
		else{
			throw new RuntimeException("unknown softmax type: " + setup.sequenceType);
		}
		return softmax;
	}
	
	public int getInputDim(){
		return setup.inputDim;
	}
	
	public static void main(String[] args){
		String[] inputWords = new String[]{"0","1","2","3","4","5","6","7","8","9",
				"10","11","12","13","14","15","16","17","18","19",
				"20","21","22","23","24","25","26","27","28","29",
				"30","31","32","33","34","35","36","37","38","39",
				"40","41","42","43","44","45","46","47","48","49",
				"50","51","52","53","54","55","56","57","58","59",
		"60"};
		String[] outputWords = new String[]{"zero","one","two","three","four","five","six","seven","eight","nine",
				"ten","elven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen",
				"twenty","twenty one","twenty two","twenty three","twenty four","twenty five","twenty six","twenty seven","twenty eight","twenty nine",
				"thirty","thirty one","thirty two","thirty three","thirty four","thirty five","thirty six","thirty seven","thirty eight","thirty nine",
				"fourty","fourty one","fourty two","fourty three","fourty four","fourty five","fourty six","fourty seven","fourty eight","fourty nine",
				"fifty","fifty one","fifty two","fifty three","fifty four","fifty five","fifty six","fifty seven","fifty eight","fifty nine",
		"sixth"};
		
		String[] testWords = new String[]{"60","61","62","63","64","65","66","67","68","69","70","71","71","72"};

		Vocab inputVocab = new Vocab();
		for(String input : inputWords){
			inputVocab.addWordToVocab(input,100);			
		}
		inputVocab.sortVocabByCount();
		inputVocab.generateHuffmanCodes();
		WordRepresentationSetup inputSetup = new WordRepresentationSetup(inputVocab, 50, 50, 100);
		inputSetup.addSequenceExtractor(new CharSequenceExtractor());
		WordRepresentationLayer inputRepresentation = new WordRepresentationLayer(inputSetup);

		Vocab outputVocab = new Vocab();
		for(String output : outputWords){
			outputVocab.addWordToVocab(output,100);			
		}
		outputVocab.sortVocabByCount();
		outputVocab.generateHuffmanCodes();
		
		OutputWordRepresentationSetup outputSetup = new OutputWordRepresentationSetup(outputVocab, inputRepresentation.getOutputDim(), 50, 100, "<unk>");
		outputSetup.sequenceType = "character";
		OutputWordRepresentationLayer softmaxObjectiveChar = new OutputWordRepresentationLayer(outputSetup);

		for(int epoch = 0; epoch < 10000; epoch++){
			for(int i = 0; i < inputWords.length; i++){
				String input = inputWords[i];
				String output = outputWords[i];
				GraphInference inference = new GraphInference(0, true);
				DenseNeuronArray inputRep = new DenseNeuronArray(inputRepresentation.getOutputDim());
				inference.addNeurons(inputRep);
				inference.addMapping(new OutputMappingStringToDense(input, inputRep, inputRepresentation));
				StringNeuronArray outputObjective = new StringNeuronArray();
				outputObjective.setExpected(output);
				inference.addNeurons(outputObjective);
				inference.addMapping(new OutputMappingDenseToString(inputRep, outputObjective, softmaxObjectiveChar));

				inference.init();
				inference.forward();
				inference.backward();
				inference.commit(0);
				
			}
			
			if(epoch % 10 == 0){
				for(int i = 0; i < inputWords.length; i++){
					String input = inputWords[i];
					String output = outputWords[i];
					GraphInference inference = new GraphInference(0, true);
					DenseNeuronArray inputRep = new DenseNeuronArray(inputRepresentation.getOutputDim());
					inference.addNeurons(inputRep);
					inference.addMapping(new OutputMappingStringToDense(input, inputRep, inputRepresentation));
					StringNeuronArray outputObjective = new StringNeuronArray();
					outputObjective.setExpected(output);
					inference.addNeurons(outputObjective);
					inference.addMapping(new OutputMappingDenseToString(inputRep, outputObjective, softmaxObjectiveChar));

					inference.init();
					inference.forward();
					System.err.println(input + " -> " + outputObjective.getOutput() + "(" + output + ")");
				}
				for(int i = 0; i < testWords.length; i++){
					String input = testWords[i];
					GraphInference inference = new GraphInference(0, false);
					DenseNeuronArray inputRep = new DenseNeuronArray(inputRepresentation.getOutputDim());
					inference.addNeurons(inputRep);
					inference.addMapping(new OutputMappingStringToDense(input, inputRep, inputRepresentation));
					inference.init();
					inference.forward();

					 TopNList<String> output = softmaxObjectiveChar.getTopN(inputRep, 5);
					System.err.println(input + " -> " + output);
				}

			}
		}
	}
}
