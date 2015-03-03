package vocab;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Scanner;
import java.util.Map.Entry;

import util.AliasMethod;

public class VocabWithHuffmanTree {

	public final static String EOS = "</s>";
	public final static int MAXCODELEN = 100;
	HashMap<String, WordEntry> hash = new HashMap<String, WordEntry>();
	ArrayList<WordEntry> words = new ArrayList<WordEntry>();
	long tokens = 0;
	int types = 0;
	int minOccur = 5;
	int maxTokens = -1;

	HashMap<Integer, Integer> numberOfwordsByLength = new HashMap<Integer, Integer>();
	int maxLenght = 0;

	AliasMethod sampling;	

	public void loadFromCountFile(String file){
		try {
			Scanner reader = new Scanner(new File(file));
			while(reader.hasNext()){
				String word = reader.next();
				int count = Integer.parseInt(reader.next());
				addWordToVocab(word);
				words.get(words.size()-1).count = count;
			}
			sortVocabByCount();
			generateHuffmanCodes();
			System.err.println("Vocab size = " + words.size());
			reader.close();
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}

	}

	public void saveToCountFile(String file){
		try {
			PrintStream out = new PrintStream(new File(file));
			for(WordEntry word : words){
				out.println(word.word + " " + word.count);
			}
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}

	public void saveVocab(PrintStream out){
		try {
			out.println(tokens);
			out.println(types);
			out.println(minOccur);
			out.println(maxTokens);
			out.println(maxLenght);
			for(WordEntry word : words){
				word.save(out);
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public void saveVocab(String file){
		try {
			PrintStream out = new PrintStream(new File(file));
			out.println(tokens);
			out.println(types);
			out.println(minOccur);
			out.println(maxTokens);
			out.println(maxLenght);
			for(WordEntry word : words){
				word.save(out);
			}
			out.close();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public void loadVocab(String file){
		Scanner reader;
		try {
			reader = new Scanner(new File(file));
			tokens = Integer.parseInt(reader.nextLine());
			types = Integer.parseInt(reader.nextLine());
			minOccur = Integer.parseInt(reader.nextLine());
			maxTokens = Integer.parseInt(reader.nextLine());
			for(int i = 0; i < types; i++){
				WordEntry wordEntry = new WordEntry("");
				wordEntry.load(reader);
				words.add(wordEntry);
				hash.put(wordEntry.word, wordEntry);
			}
			reader.close();
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);
		}
	}

	public static VocabWithHuffmanTree loadVocab(BufferedReader in){
		try {
			VocabWithHuffmanTree vocab = new VocabWithHuffmanTree();
			vocab.tokens = Integer.parseInt(in.readLine());
			vocab.types = Integer.parseInt(in.readLine());
			vocab.minOccur = Integer.parseInt(in.readLine());
			vocab.maxTokens = Integer.parseInt(in.readLine());
			vocab.maxLenght = Integer.parseInt(in.readLine());
			for(int i = 0; i < vocab.types; i++){
				WordEntry wordEntry = WordEntry.load(in);
				vocab.words.add(wordEntry);
				vocab.hash.put(wordEntry.word, wordEntry);
			}
			return vocab;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public void addWordToVocab(String word){
		addWordToVocab(word,1);
	}	

	public void addWordToVocab(String word, int count){
		WordEntry wordObj = null;		
		if(!hash.containsKey(word)){
			wordObj = new WordEntry(word);
			hash.put(word, wordObj);
			words.add(wordObj);
			wordObj.id = types;
			types++;
		}
		else{
			wordObj = hash.get(word);
		}		
		if(!numberOfwordsByLength.containsKey(word.length())){
			numberOfwordsByLength.put(word.length(), 0);
		}
		numberOfwordsByLength.put(word.length(), numberOfwordsByLength.get(word.length()) + count);
		if(word.length() > maxLenght){
			maxLenght = word.length();
		}
		wordObj.count+=count;
		tokens+=count;
	}	

	public void reduceVocab(){
		ArrayList<WordEntry> toAdd = new ArrayList<WordEntry>();
		final WordEntry eof = hash.get(EOS);
		System.err.println("adding entries to remove");
		for(Entry<String, WordEntry> entry : hash.entrySet()){
			if(entry.getValue() != eof && entry.getValue().count < minOccur){

			}
			else{
				toAdd.add(entry.getValue());
			}
		}
		System.err.println("removing");
		words.clear();
		hash.clear();
		for(WordEntry entry : toAdd){
			words.add(entry);
			hash.put(entry.word, entry);
		}
		types=words.size();
	}

	public void sortVocabByCount(int maxEntries){
		sortVocabByCount();
		while(words.size() > maxEntries){
			WordEntry entry = words.remove(words.size()-1);
			hash.remove(entry.word);
		}
		this.types = words.size();
		System.err.println("reducing vocab to " + words.size() + " entries");
	}

	public void sortVocabByCount(){
		final WordEntry eof = hash.get(EOS);
		Collections.sort(words, new Comparator<WordEntry>() {
			@Override
			public int compare(WordEntry o1, WordEntry o2) {
				if(o1 == eof){
					return -1; // make sure eof is first
				}
				if(o2 == eof){
					return 1; // make sure eof is first
				}
				if(o1.count>o2.count) return -1;
				if(o1.count<o2.count) return 1;
				return 0;
			}
		});
		for(int i = 0; i < words.size(); i++){
			words.get(i).id = i;
		}
	}

	//make sure to run sortVocab first
	public void generateHuffmanCodes(){
		int[] parentIndexes = new int[words.size() * 2]; 
		int[] count = new int[words.size() * 2];
		int[] binary = new int[words.size() * 2];
		int min1index, min2index;

		for(int i = 0; i < words.size(); i++){
			count[i] = words.get(i).count;
			if(count[i] == 0){
				count[i] = 1;
			}
			count[i+words.size()] = Integer.MAX_VALUE;
		}

		int lowestIndexLeft = words.size() - 1;
		int lowestIndexRight = words.size();

		for(int i = 0; i < words.size() - 1; i++){
			if(lowestIndexLeft >= 0){
				if(count[lowestIndexLeft] < count[lowestIndexRight]){
					min1index = lowestIndexLeft;
					lowestIndexLeft--;
				}
				else{
					min1index = lowestIndexRight;
					lowestIndexRight++;
				}
			}
			else{
				min1index = lowestIndexRight;
				lowestIndexRight++;
			}
			if(lowestIndexLeft >= 0){
				if(count[lowestIndexLeft] < count[lowestIndexRight]){
					min2index = lowestIndexLeft;
					lowestIndexLeft--;
				}
				else{
					min2index = lowestIndexRight;
					lowestIndexRight++;
				}
			}
			else{
				min2index = lowestIndexRight;
				lowestIndexRight++;
			}
			count[i + words.size()] = count[min1index] + count[min2index];
			parentIndexes[min1index] = i + words.size();
			parentIndexes[min2index] = i + words.size();
			binary[min2index] = 1;
		}

		for (int i = 0; i < words.size(); i++){
			int index = i;
			int codeBit = 0;
			ArrayList<Integer> code = new ArrayList<Integer>();
			ArrayList<Integer> points = new ArrayList<Integer>();
			WordEntry entry = words.get(i);
			while(index < words.size() * 2 - 2){
				code.add(binary[index]);
				points.add(index);
				index = parentIndexes[index];
				codeBit++;
			}

			entry.code = new int[codeBit];
			entry.point = new int[codeBit+1];
			entry.point[0] = words.size() - 2;
			for(int j = 0; j < codeBit; j++){
				entry.code[codeBit - j - 1] = code.get(j);
				entry.point[codeBit - j] = points.get(j) - words.size();
			}

			if(codeBit == MAXCODELEN){
				throw new RuntimeException("for some reason "+MAXCODELEN+" bits were used for the code");
			}
		}
	}

	public void printWordCounts(){
		System.err.println("hash table counts");
		for(Entry<String, WordEntry> entry : hash.entrySet()){
			System.err.println(entry.getKey() + " - " + entry.getValue().count);
		}

		System.err.println();
		System.err.println("word list");
		for(WordEntry entry : words){
			System.err.println(entry.word + " - " + entry);			
		}
	}	

	public WordEntry getEntry(String word){
		if(hash.containsKey(word)){
			return hash.get(word);
		}
		return null;
	}

	public WordEntry[] getEntriesFromArray(String[] wordBlock, int blockSize) {
		WordEntry[] ret = new WordEntry[blockSize];
		for(int i = 0; i<blockSize;i++){
			ret[i] = getEntry(wordBlock[i]);
		}
		return ret;
	}

	public WordEntry[] getEntriesFromArray(char[] wordBlock, int blockSize) {
		WordEntry[] ret = new WordEntry[blockSize];
		for(int i = 0; i<blockSize;i++){
			ret[i] = getEntry(String.valueOf(wordBlock[i]));
		}
		return ret;
	}

	public int[] getEntryIdsFromArray(String[] wordBlock, int blockSize) {
		int[] ret = new int[blockSize];
		for(int i = 0; i<blockSize;i++){
			WordEntry entry = getEntry(wordBlock[i]);
			if(entry!= null){
				ret[i] = entry.id;
			}
			else{
				ret[i] = -1;
			}
		}
		return ret;
	}

	public WordEntry getEntryFromId(int i) {
		return words.get(i);
	}

	public int getTypes() {
		return types;
	}

	public void setMinOccur(int minOccur) {
		this.minOccur = minOccur;
	}

	public long getTokens() {
		return tokens;
	}

	public void learnCharacterVocabFromWordVocab(VocabWithHuffmanTree wordVocab){
		addWordToVocab(EOS);
		for(WordEntry word : wordVocab.words){
			for(char c : word.word.toCharArray()){
				addWordToVocab(""+c);				
			}
		}
		reduceVocab();
		sortVocabByCount();
		generateHuffmanCodes();
		System.err.println("Vocab size = " + words.size());		
	}

	public ArrayList<WordEntry> getWords() {
		return words;
	}

	public int getEOSIndex(){
		return 0;
	}

	public void setMaxTokens(int maxTokens) {
		this.maxTokens = maxTokens;
	}

	public double getWordProb(String word){
		WordEntry wordEntry = getEntry(word);
		if(wordEntry == null) return 0;
		return wordEntry.count / ((double)tokens);
	}

	public double getLargerWordProb(String word) {
		WordEntry wordEntry = getEntry(word);
		if(wordEntry == null) return 0;
		double largerCount = 0;
		for(int i = word.length() + 1; i <= maxLenght; i++){
			if(numberOfwordsByLength.containsKey(i)){
				largerCount += numberOfwordsByLength.get(i);
			}
		}
		return wordEntry.count / (wordEntry.count + largerCount);
	}

	public void printTopWords(int top) {
		for(int i = 0; i < top; i++){
			System.err.println(i + ":" + getEntryFromId(i));
		}
	}

	public String genNewWord(){
		String ret = "thiswordismineallllllllllmine";
		while(hash.containsKey(ret)){
			ret+="0";
		}
		return ret;
	}

	public WordEntry getRandomEntry() {
		return words.get((int)(Math.random() * getTypes()));
	}

	public WordEntry getRandomEntryByCount() {
		if(sampling == null){
			LinkedList<Double> wordCounts = new LinkedList<Double>();
			for(int i = 0; i < getTypes(); i++){
				wordCounts.addLast((double)words.get(i).getCount());				
			}
			sampling = new AliasMethod(wordCounts);
		}		
		return words.get(sampling.next());
	}
}
