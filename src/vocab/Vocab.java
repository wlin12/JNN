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
import util.IOUtils;
import util.MathUtils;
import util.SerializeUtils;
import vocab.ArbitraryHuffman.Node;

public class Vocab {

	public final static String EOS = "</s>";
	public final static int MAXCODELEN = 100;
	HashMap<String, WordEntry> hash = new HashMap<String, WordEntry>();
	ArrayList<WordEntry> words = new ArrayList<WordEntry>();
	long tokens = 0;
	int types = 0;
	int minOccur = 5;
	int maxTokens = -1;

	HashMap<Integer, Long> numberOfwordsByLength = new HashMap<Integer, Long>();
	int maxLenght = 0;

	AliasMethod sampling;
	
	int numberOfHuffmanNodes = 0;
	int[][] huffmanNodeChildren;
	int[] nodeNumberOfChildren;
	int[] nodeLevel;
	long[] nodeCount;

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
			out.println(numberOfHuffmanNodes);
			if(numberOfHuffmanNodes > 0){
				SerializeUtils.saveIntMatrix(huffmanNodeChildren, out);
				SerializeUtils.saveIntArray(nodeNumberOfChildren, out);
				SerializeUtils.saveIntArray(nodeLevel, out);
				SerializeUtils.saveLongArray(nodeCount, out);
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

	public static Vocab loadVocab(BufferedReader in){
		try {
			Vocab vocab = new Vocab();
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
			vocab.numberOfHuffmanNodes = Integer.parseInt(in.readLine());
			if(vocab.numberOfHuffmanNodes > 0){
				vocab.huffmanNodeChildren = SerializeUtils.loadIntMatrix(in);
				vocab.nodeNumberOfChildren = SerializeUtils.loadIntArray(in);
				vocab.nodeLevel = SerializeUtils.loadIntArray(in);
				vocab.nodeCount = SerializeUtils.loadLongArray(in);				
			}
			return vocab;
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	public void addWordToVocab(String word){
		addWordToVocab(word,1);
	}	

	public void addWordToVocab(String word, long count){
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
			numberOfwordsByLength.put(word.length(), 0l);
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
		long[] count = new long[words.size() * 2];
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

		numberOfHuffmanNodes = words.size()-1; 
		huffmanNodeChildren = new int[numberOfHuffmanNodes][2];
		nodeNumberOfChildren = new int[numberOfHuffmanNodes];
		nodeLevel = new int[numberOfHuffmanNodes];
		nodeCount = new long[numberOfHuffmanNodes];

		for(int i = 0; i < numberOfHuffmanNodes; i++){
			for(int j = 0; j < 2; j++){
				huffmanNodeChildren[i][j] = -1;
			}
			nodeNumberOfChildren[i] = 2;
		}
		
		for(int i = 0; i < numberOfHuffmanNodes; i++){
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
			huffmanNodeChildren[i][0] = min1index;
			huffmanNodeChildren[i][1] = min2index;
			int node1Level = 0;
			if(min1index >= words.size()){
				node1Level = nodeLevel[min1index-words.size()];
				nodeCount[i]+=nodeCount[min1index-words.size()];
			}
			else{
				nodeCount[i]+=getEntryFromId(min1index).count;
			}
			int node2Level = 0;
			if(min2index >= words.size()){
				node2Level = nodeLevel[min2index-words.size()];
				nodeCount[i]+=nodeCount[min2index-words.size()];
			}
			else{
				nodeCount[i]+=getEntryFromId(min2index).count;				
			}
			nodeLevel[i] = Math.max(node1Level, node2Level)+1;
			binary[min2index] = 1;

			//			binary[min1index] = 0;
		}

		for (int i = 0; i < words.size(); i++){
			int index = i;
			int codeBit = 0;
			ArrayList<Integer> code = new ArrayList<Integer>();
			ArrayList<Integer> points = new ArrayList<Integer>();
			WordEntry entry = words.get(i);
			while(index < words.size() + numberOfHuffmanNodes - 1){
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

	public void generateHuffmanCodesForNAryTree(int nary){
		int[] parentIndexes = new int[words.size() * 2]; 
		long[] count = new long[words.size() * 2];
		int[] codeAtPoint = new int[words.size() * 2];
		int[] minIndexes = new int[nary];

		for(int i = 0; i < words.size(); i++){
			count[i] = words.get(i).count;
			if(count[i] == 0){
				count[i] = 1;
			}
			count[i+words.size()] = Integer.MAX_VALUE;
		}

		int lowestIndexLeft = words.size() - 1;
		int lowestIndexRight = words.size();

		numberOfHuffmanNodes = (words.size()-1) / (nary-1); 
		
		if((words.size()-1) % (nary-1) > 0){
			numberOfHuffmanNodes++;
		}
		huffmanNodeChildren = new int[numberOfHuffmanNodes][nary];
		nodeNumberOfChildren = new int[numberOfHuffmanNodes];
		nodeLevel = new int[numberOfHuffmanNodes];
		nodeCount = new long[numberOfHuffmanNodes];
		for(int i = 0; i < numberOfHuffmanNodes; i++){
			for(int j = 0; j < nary; j++){
				huffmanNodeChildren[i][j] = -1;
			}
		}
		
		for(int i = 0; i < numberOfHuffmanNodes; i++){			
			for(int n = 0; n < nary; n++){
				if(lowestIndexLeft >= 0){
					if(count[lowestIndexLeft] < count[lowestIndexRight]){
						minIndexes[n] = lowestIndexLeft;
						lowestIndexLeft--;
					}
					else{
						minIndexes[n] = lowestIndexRight;
						lowestIndexRight++;
					}
				}
				else{
					minIndexes[n] = lowestIndexRight;
					lowestIndexRight++;
				}				
			}	
			count[i + words.size()] = 0;
			int level = 0;
			for(int n = 0; n < nary; n++){					
				if(minIndexes[n]<words.size()+numberOfHuffmanNodes-1){
					count[i + words.size()]+=count[minIndexes[n]];
					parentIndexes[minIndexes[n]] = i + words.size();
					codeAtPoint[minIndexes[n]] = n;
					if(huffmanNodeChildren[i][n]!=-1){
						throw new RuntimeException("repeated node found");
					}
					huffmanNodeChildren[i][n] = minIndexes[n];
					nodeNumberOfChildren[i]++;					
					if(minIndexes[n] >= words.size()){
						if(nodeLevel[minIndexes[n]-words.size()] > level){
							level = nodeLevel[minIndexes[n]-words.size()];
						}
						nodeCount[i] += nodeCount[minIndexes[n]-words.size()];
					}
					else{
						nodeCount[i] += getEntryFromId(minIndexes[n]).count;
					}					
				}
			}
			nodeLevel[i] = level+1;
			
		}

		int totalNumberOfNodes = words.size() + numberOfHuffmanNodes;
		for (int i = 0; i < words.size(); i++){
			int index = i;
			int codeBit = 0;
			ArrayList<Integer> code = new ArrayList<Integer>();
			ArrayList<Integer> points = new ArrayList<Integer>();
			WordEntry entry = words.get(i);
			while(index < totalNumberOfNodes - 1){
				code.add(codeAtPoint[index]);
				points.add(index);
				index = parentIndexes[index];
				codeBit++;
			}

			entry.code = new int[codeBit];
			entry.point = new int[codeBit+1];
			entry.point[0] = numberOfHuffmanNodes-1;
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

	public void learnCharacterVocabFromWordVocab(Vocab wordVocab){
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
	
	public int[] getHuffmanNodeChildren(int nodeId){
		return huffmanNodeChildren[nodeId];
	}

	public int getHuffmanNodeNumberOfChildren(int nodeId){
		return nodeNumberOfChildren[nodeId];
	}
	
	public int getNumberOfHuffmanNodes() {
		return numberOfHuffmanNodes;
	}
	
	public void printHuffmanTree(){
		int maxLevel = nodeLevel[numberOfHuffmanNodes-1]+1;
		LinkedList<Integer>[] nodesPerLevel = new LinkedList[maxLevel];
		for(int i = maxLevel-1; i >= 0; i--){
			nodesPerLevel[i] = new LinkedList<Integer>();
		}
		for(int i = 0; i < getTypes(); i++){
			nodesPerLevel[0].add(i);
		}
		for(int i = 0; i < numberOfHuffmanNodes; i++){
			nodesPerLevel[nodeLevel[i]].addFirst(i);
		}

//		long score = 0;
//		for(Node node : nodes){
//			if(node != null)
//			nodesPerLevel[node.level-1].addLast(node);
//			score+=node.operationsForSoftmax();
//		}
//		
//		System.err.println("------------------");			
//		System.err.println("---score: " + score + " --------------");
//		System.err.println("------------------");
		
		String[] outputPerLevel = new String[maxLevel];
		int[] leftOffsetPerNode = new int[getTypes() + numberOfHuffmanNodes];
		int[] rightOffsetPerNode = new int[getTypes() + numberOfHuffmanNodes];
		for(int i = 0; i < maxLevel; i++){
			String output = "";
			for(int node : nodesPerLevel[i]){
				if(i == 0){
					output += "    ";
					leftOffsetPerNode[node] = output.length();
					output+= node+"-"+getEntryFromId(node).word + "(" + getEntryFromId(node).count + ")";
					rightOffsetPerNode[node] = output.length();
					output += "    ";
				}
				else{
					int[] lefts = new int[nodeNumberOfChildren[node]];
					int[] rights = new int[nodeNumberOfChildren[node]];
					for(int j = 0 ; j < nodeNumberOfChildren[node]; j++){
						lefts[j] = leftOffsetPerNode[huffmanNodeChildren[node][j]];
						rights[j] = rightOffsetPerNode[huffmanNodeChildren[node][j]];
					}
					
					int left = MathUtils.min(lefts);
					int right = MathUtils.max(rights);
//					String nodeStr = node.children.getFirst().id + "-" + node.toString() + "-" + node.children.getLast().id;
					String nodeStr = (node + words.size()) + "("+nodeCount[node] + ")";
					leftOffsetPerNode[node + words.size()] = left + ((right - left - nodeStr.length())/2);
					rightOffsetPerNode[node + words.size()] = left + ((right - left + nodeStr.length())/2);
					while(output.length()<left){
						output+=" ";
					}
					output+="|";
					while(output.length()<leftOffsetPerNode[node + words.size()]-1){
						output+="-";
					}						
					output+=nodeStr;
					while(output.length()<right-1){
						output+="-";
					}		
					output+="|";						
				}
			}
			outputPerLevel[i] = output;
		}
		for(int i = maxLevel-1; i >= 0; i--){
			System.err.println(outputPerLevel[i]);
		}
		System.err.println("------------------");
	}
	
	public static void main(String[] args){
		Vocab vocab = new Vocab();
		vocab.addWordToVocab("p", 11);
		vocab.addWordToVocab("tiger", 10);
		vocab.addWordToVocab("monkey", 7);
		vocab.addWordToVocab("cat", 3);
		vocab.addWordToVocab("dog", 3);
		vocab.addWordToVocab("mouse", 2);
		vocab.addWordToVocab("spider", 1);
		vocab.addWordToVocab("bat", 1);
		vocab.addWordToVocab("rhino", 1);
//		vocab.addWordToVocab("whale", 1);
//		vocab.addWordToVocab("eagle", 1);
//		vocab.addWordToVocab("dove", 1);
//		vocab.addWordToVocab("rat", 1);
//		vocab.addWordToVocab("pidgeon", 1);
//		vocab.addWordToVocab("ant", 1);
//		vocab.addWordToVocab("python", 1);
		vocab.sortVocabByCount();
		vocab.generateHuffmanCodesForNAryTree(2);
		vocab.printWordCounts();
		vocab.printHuffmanTree();
		
		String tmpFile = "/tmp/file";
		PrintStream out = IOUtils.getPrintStream(tmpFile);
		vocab.saveVocab(out);
		out.close();
		
		BufferedReader in = IOUtils.getReader(tmpFile);
		Vocab loaded = Vocab.loadVocab(in);
		loaded.printHuffmanTree();
	}

	public int[] getHuffmanNodesForEntry(WordEntry expectedEntry) {
		int[] ret = new int[expectedEntry.code.length];
		for(int i = 0; i < ret.length; i++){
			ret[i] = expectedEntry.point[i];
		}
		return ret;
	}

	public int getInitialHuffmanNode() {
		return numberOfHuffmanNodes-1;
	}

	public boolean isHuffmanNode(int id) {
		return id >= getTypes();
	}

	public int idToHuffmanNode(int id) {
		return id - getTypes();
	}
	
}
