package jnn.functions.nlp.app.pretraining;

import java.util.HashMap;

public class WordPairs {
	public HashMap<String, HashMap<String, Integer>> wordPairs = new HashMap<String, HashMap<String,Integer>>();	
	
	public void addWordPair(String input, String output) {
		if(!wordPairs.containsKey(input)){
			wordPairs.put(input, new HashMap<String, Integer>());
		}
		HashMap<String, Integer> outputWords = wordPairs.get(input);
		if(!outputWords.containsKey(output)){
			outputWords.put(output, 1);
		}
		else{
			outputWords.put(output, 1+outputWords.get(output));
		}
	}

	public HashMap<String, Integer> getWordsFor(String input) {
		if(!wordPairs.containsKey(input)){
			return new HashMap<String, Integer>();
		}
		else{
			return wordPairs.get(input);
		}
	}

	public void remove(String s) {
		if(wordPairs.containsKey(s)){
			wordPairs.remove(s);
		}
	}
}
