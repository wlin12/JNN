package jnn.functions.nlp.aux.input;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

import jnn.functions.nlp.aux.readers.ConllReader;
import jnn.functions.nlp.aux.readers.NpsReader;
import jnn.functions.nlp.aux.readers.ParallelFormatReader;
import jnn.functions.nlp.aux.readers.LabelledReaderCallback;
import jnn.functions.nlp.aux.readers.SplitedFeaturesReader;
import jnn.functions.nlp.aux.readers.StanfordOriginalFormatReader;
import jnn.functions.nlp.aux.readers.StanfordPosReader;

public class LabelledData {

	public String file;
	public ArrayList<LabelledSentence> sentences;

	public LabelledData() {
		String file = "null";
		sentences = new ArrayList<LabelledSentence>();
	}

	public LabelledData(String file){
		this.file = file;
		init(file, "text");
	}

	public LabelledData(String file, String type){
		this.file = file;
		init(file, type);
	}

	public void init(String file, String type){
		sentences = new ArrayList<LabelledSentence>();
		if(type.toLowerCase().startsWith("conll")){
			int word = Integer.parseInt(type.split("-")[1]);
			int pos = Integer.parseInt(type.split("-")[2]);
			ConllReader.read(file,word,pos, new LabelledReaderCallback() {

				@Override
				public void cb(LabelledSentence sent) {
					if(sent.tokens.length > 0){
						sentences.add(sent);		
					}
				}
			});
		}
		else if(type.toLowerCase().equals("nps")){
			NpsReader.read(file, new LabelledReaderCallback() {

				@Override
				public void cb(LabelledSentence sent) {
					if(sent.tokens.length > 0){
						sentences.add(sent);		
					}				
				}
			});
		}
		else if(type.toLowerCase().equals("stanford")){
			StanfordOriginalFormatReader.read(file, new LabelledReaderCallback() {

				@Override
				public void cb(LabelledSentence sent) {
					if(sent.tokens.length > 0){
						sentences.add(sent);				
					}

				}
			});
		}
		else if(type.toLowerCase().equals("parallel")){
			ParallelFormatReader.read(file, new LabelledReaderCallback() {

				@Override
				public void cb(LabelledSentence sent) {
					if(sent.tokens.length > 0){
						sentences.add(sent);				
					}

				}
			});
		}
		else if(type.toLowerCase().equals("splited")){
			SplitedFeaturesReader.read(file, new LabelledReaderCallback() {

				@Override
				public void cb(LabelledSentence sent) {
					if(sent.tokens.length > 0){
						sentences.add(sent);				
					}

				}
			});
		}
		else{
			StanfordPosReader.read(file, new LabelledReaderCallback() {

				@Override
				public void cb(LabelledSentence sent) {
					if(sent.tokens.length > 0){
						sentences.add(sent);		
					}
				}
			});
		}
	}

	public ArrayList<LabelledSentence> getSentences() {
		return sentences;
	}

	public int getSize(){
		return sentences.size();
	}

	public void scramble() {
		Collections.shuffle(sentences);
	}

	public double accuracy(LabelledData refDataset) {
		if(refDataset.sentences.size() != sentences.size()){
			throw new RuntimeException("number of sentences does not match");
		}
		int matches = 0;
		int words = 0;
		for(int i = 0 ; i < sentences.size(); i++){
			String[] hyp = sentences.get(i).tags;
			String[] ref = refDataset.sentences.get(i).tags;
			for(int j = 0; j < hyp.length; j++){
				if(hyp[j].equals(ref[j])){
					matches++;
				}
				words++;
			}
		}
		return matches/(double)words;
	}

	public double oovAccuracy(LabelledData refDataset, LabelledData trainDataset){
		HashSet<String> trainWords = new HashSet<String>();
		for(int i = 0 ; i < trainDataset.sentences.size(); i++){
			String[] tokens = trainDataset.sentences.get(i).tokens;
			for(int j = 0; j < tokens.length; j++){
				trainWords.add(tokens[j]);
			}
		}
		if(refDataset.sentences.size() != sentences.size()){
			throw new RuntimeException("number of sentences does not match");
		}
		int matches = 0;
		int words = 0;
		for(int i = 0 ; i < sentences.size(); i++){
			String[] hyp = sentences.get(i).tags;
			String[] ref = refDataset.sentences.get(i).tags;
			for(int j = 0; j < hyp.length; j++){
				if(!trainWords.contains(ref[j])){
					if(hyp[j].equals(ref[j])){
						matches++;
					}
					words++;
				}
			}
		}
		return matches/(double)words;
	}
}
