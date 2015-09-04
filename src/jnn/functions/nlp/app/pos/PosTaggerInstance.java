package jnn.functions.nlp.app.pos;

import java.util.ArrayList;

import jnn.functions.nlp.aux.input.InputSentence;
import jnn.functions.nlp.aux.input.LabelledSentence;
import jnn.functions.nlp.labeling.WordTaggingLayer;
import jnn.mapping.Mapping;
import jnn.mapping.OutputMappingStringArrayToStringArray;
import jnn.neuron.StringNeuronArray;
import jnn.training.GraphInference;

public class PosTaggerInstance {

	int id;
	WordTaggingLayer taggerLayer;

	LabelledSentence[] batchSents;
	double norm;
	
	ArrayList<LabelledSentence> sentences = new ArrayList<LabelledSentence>();
	ArrayList<LabelledSentence> hypothesis = new ArrayList<LabelledSentence>();

	//debug only
	GraphInference subinference;
	
	public PosTaggerInstance(int id, LabelledSentence[] batchSents, WordTaggingLayer taggerLayer, double norm) {
		super();
		this.id = id;
		this.taggerLayer = taggerLayer;
		this.batchSents = batchSents;
		this.norm = norm;
	}
	
	public StringNeuronArray[] buildNetwork(InputSentence sent, GraphInference inference, String[] ref){
		String[] input = sent.tokens;
		
		StringNeuronArray[] tags = StringNeuronArray.asArray(input.length);
		inference.addNeurons(tags);
		Mapping map = new OutputMappingStringArrayToStringArray(input, tags, taggerLayer);
		inference.addMapping(map);
		subinference = map.getSubInference();
		if(ref!=null){
			StringNeuronArray.setExpectedArray(ref, tags);
		}
		inference.init();
		
		inference.forward();		
		return tags;
	}

	public void train(){
		for(int i = 0; i < batchSents.length; i++){
			GraphInference inference = new GraphInference(id, true);
			inference.setNorm(norm);
			StringNeuronArray[] predicted = buildNetwork(batchSents[i], inference, batchSents[i].tags);			
			inference.backward();
			LabelledSentence sent = getPrediction(batchSents[i], predicted);
			sentences.add(batchSents[i]);
			hypothesis.add(sent);
		}
	}
	
	public void test(){
		for(int i = 0; i < batchSents.length; i++){
			GraphInference inference = new GraphInference(id, false);
			inference.setNorm(norm);
			StringNeuronArray[] predicted = buildNetwork(batchSents[i], inference, null);			
			LabelledSentence sent = getPrediction(batchSents[i], predicted);
			sentences.add(batchSents[i]);
			hypothesis.add(sent);
		}
	}
	
	public static LabelledSentence getPrediction(InputSentence input, StringNeuronArray[] predictions){		
		LabelledSentence ret = new LabelledSentence(input.original, input.tokens, StringNeuronArray.toStringArray(predictions));
		return ret;
	}
	
	public ArrayList<LabelledSentence> getSentences() {
		return sentences;
	}
	
	public ArrayList<LabelledSentence> getHypothesis() {
		return hypothesis;
	}
}
