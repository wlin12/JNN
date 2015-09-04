package jnn.functions.nlp.words.features;

import vocab.Vocab;

public class OutputWordRepresentationSetup {

	public Vocab existingWords;
	public int inputDim;
	public int characterDim = 50;
	public int stateDim = 150;
	public String sequenceType = "word";
	public int beam = 1;
	public String UNK;
	public int negativeSamplingRate = 100;
	
	public OutputWordRepresentationSetup(Vocab existingWords, int inputDim,
			int characterDim, int stateDim, String UNK) {
		super();
		this.existingWords = existingWords;
		this.inputDim = inputDim;
		this.characterDim = characterDim;
		this.UNK = UNK;
		this.stateDim = stateDim;
	}
		
}
