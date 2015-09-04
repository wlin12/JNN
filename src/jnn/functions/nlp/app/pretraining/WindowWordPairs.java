package jnn.functions.nlp.app.pretraining;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Set;

import util.TopNList;
import vocab.Vocab;

public class WindowWordPairs {
	int windowSize;
	WordPairs[] pairsPerWindow;
	HashMap<String, Integer> counts = new HashMap<String, Integer>();
	
	public WindowWordPairs(int windowSize) {
		this.windowSize = windowSize;
		pairsPerWindow = new WordPairs[windowSize*2];
		for(int i = 0; i < windowSize*2; i++){
			pairsPerWindow[i] = new WordPairs();
		}
	}
	
	public int getCount(String word){
		return counts.get(word);
	}
	
	public Set<String> getWords(){
		return counts.keySet();
	}
	
	public int addWord(String[] sentence, int center, boolean[] validWords){
		
		String word = sentence[center];
		for(int w = -windowSize; w <= windowSize; w++){
			if(w != 0){
				int pos = center + w;
				if(pos > 0 && pos < sentence.length){
					int index = w + windowSize;
					if(w>0){
						index--;
					}
					if(validWords[pos]){
						pairsPerWindow[index].addWordPair(word, sentence[pos].toLowerCase());					
					}
				}
			}
		}
		if(!counts.containsKey(word)){
			counts.put(word, 1);
			return 1;
		}
		else{
			int val = counts.get(word);
			val++;
			counts.put(word, val);
			return val;
		}
		
		
	}

	public void remove(String s) {
		counts.remove(s);
		for(int i = 0; i < windowSize*2; i++){
			pairsPerWindow[i].remove(s);
		}
	}
	
	public String readFromCountFile(BufferedReader reader){
		try {
			String line = reader.readLine();
			String[] lineArray = line.split("\t");
			String source = lineArray[0];
			String target = lineArray[1];
			int windowPos = Integer.parseInt(lineArray[2]);
			int count = Integer.parseInt(lineArray[3]);
			if(!counts.containsKey(source)){
				counts.put(source, count);
			}
			else{
				counts.put(source, counts.get(source) + count);
			}
			pairsPerWindow[windowPos].addWordPair(source, target);
			return source;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public int size() {		
		return counts.size();
	}
}
