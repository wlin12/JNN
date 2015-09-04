package jnn.wordsim;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Scanner;
import java.util.SortedSet;
import java.util.Vector;

import org.apache.commons.math3.util.FastMath;

import jnn.neuron.DenseNeuronArray;
import util.MapUtils;
import util.MathUtils;
import util.PrintUtils;
import util.TopNList;
import vocab.Vocab;
import vocab.WordEntry;

public class WordVectors {

	double[][] vectors;
	double[][] cosineDistA;
	
	public WordVectors(double[][] vectors){
		this.vectors = vectors;
		this.cosineDistA = new double[vectors.length][vectors[0].length];
		for(int i = 0; i < cosineDistA.length; i++){			
			cosineDistA[i] = MathUtils.normVector(vectors[i]);			
		}
	}	

	public double cosineSimilarity(double[] a, double[] b){
		double distance = 0;
		for(int i = 0; i < a.length; i++){
			distance += a[i]*b[i];
		}
		return distance;
	}
	
	public double euclideanSimilarity(double[] a, double[] b){
		double distance = 0;
		for(int i = 0; i < a.length; i++){
			distance += (a[i]-b[i])*(a[i]-b[i]);
		}
		return -FastMath.sqrt(distance);
	}
	
	public double cosineSimilarity(int wordA, int wordB){
		return cosineSimilarity(cosineDistA[wordA], cosineDistA[wordB]);
	}
	
	public double[] getDistPerDimenstion(double[] a, double[] b){
		double[] sim = new double[a.length];
		for(int i = 0; i < a.length; i++){
			sim[i] += a[i]*b[i];
		}
		return sim;
	}
	
	public String getTopSimilarFeatures(double[] a, double[] b, int top){
		double[] dist = getDistPerDimenstion(a, b);
		HashMap<String, Double> similarities = new HashMap<String, Double>();
		for(int i = 0; i < a.length; i++){
			similarities.put(i+"", dist[i]);
		}
		String ret = "";
		LinkedList<String> bottom = MapUtils.getBottomN(similarities, top);
		for(String dim : bottom){
			ret+=dim + " ";
		}
		return ret;
	}

	public WordVectors(String file){
		try {
			Scanner reader = new Scanner(new File(file));
			int size = reader.nextInt();
			int dim = reader.nextInt();
			System.err.println(size);
			reader.nextLine();
			vectors = new double[size][dim];
			int word = 0;
			while(reader.hasNext()){
				String line = reader.nextLine();
				String[] lineArray = line.split("\\s+");
				for(int i = 1; i <= dim; i++){
					vectors[word][i-1] = Double.parseDouble(lineArray[i]);
				}
				word++;
			}
		} catch (FileNotFoundException e) {			
			e.printStackTrace();
			System.exit(0);
		}
		this.cosineDistA = new double[vectors.length][vectors[0].length];
		for(int i = 0; i < cosineDistA.length; i++){			
			double len = 0;			
			for(int j = 0; j < vectors[i].length; j++){
				len += vectors[i][j] * vectors[i][j]; 
			}
			len = Math.sqrt(len);
			for(int j = 0; j < vectors[i].length; j++){
				cosineDistA[i][j] = vectors[i][j]/len;
			}
		}
	}

	public WordVectors(DenseNeuronArray[] wordReps) {
		vectors = new double[wordReps.length][wordReps[0].size];
		for(int i = 0; i < wordReps.length; i++){
			vectors[i] = wordReps[i].copyAsArray();
		}
		this.cosineDistA = new double[vectors.length][vectors[0].length];
		for(int i = 0; i < cosineDistA.length; i++){			
			cosineDistA[i] = MathUtils.normVector(vectors[i]);			
		}
	}

	public void saveToFile(String file, Vocab vocab){
		try {
			PrintStream out = new PrintStream(new File(file));
			out.println(vectors.length + " " + vectors[0].length);
			for(int i = 0; i < vectors.length; i++){
				StringBuffer line = new StringBuffer();
				line.append(vocab.getEntryFromId(i).getWord() + " ");
				for(int j = 0; j < vectors[i].length; j++){
					line.append(vectors[i][j]);
					if(j != vectors[i].length - 1){
						line.append(" ");
					}
				}
				out.println(line);
			}
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(0);
		}
	}

	public void printTopN(String word, int topN, Vocab vocab){
		printTopN(vocab.getEntry(word).getId(), topN, vocab);
	}

	public void printTopN(String word, int topN, Vocab vocab, PrintStream out){
		printTopN(vocab.getEntry(word).getId(), topN, vocab, out);
	}

	public void printTopN(int word, int topN, Vocab vocab){
		double[] vectorForWord = cosineDistA[word];
		printTopN(vectorForWord, topN, vocab);
	}
	
	public void printTopN(int word, int topN, String[] words){
		double[] vectorForWord = cosineDistA[word];
		printTopN(vectorForWord, topN, words);
	}
	
	public void printTopN(int word, int topN, String[] words, PrintStream out){
		double[] vectorForWord = cosineDistA[word];
		printTopN(vectorForWord, topN, words,out);
	}

	public void printTopN(int word, int topN, Vocab vocab, PrintStream out){
		double[] vectorForWord = cosineDistA[word];
		printTopN(vectorForWord, topN, vocab, out);
	}

	public void printTopN(double[] embedding, int topN, Vocab vocab){
		printTopN(embedding, topN, vocab, System.err);
	}
	
	public void printTopN(double[] embedding, int topN, String[] words){
		printTopN(embedding, topN, words, System.err);
	}

	public void printTopN(double[] embedding, int topN, Vocab vocab, PrintStream out){
		embedding = MathUtils.normVector(embedding);
		TopNList<Integer> topNlist = getTopN(embedding, topN);		
		Iterator<Integer> itObj = topNlist.getObjectList().iterator();
		Iterator<Double> itDist = topNlist.getObjectScore().iterator();
		while(itObj.hasNext()){
			Integer comWord = itObj.next();
			Double comDist = itDist.next();
			out.println(vocab.getEntryFromId(comWord).getWord() + "(" + vocab.getEntryFromId(comWord).count + ")" + " -> " + comDist + " (" + getTopSimilarFeatures(embedding,cosineDistA[vocab.getEntryFromId(comWord).id],10) + ")");
		}
	}
	
	public void printTopN(double[] embedding, int topN, String[] words, PrintStream out){
		embedding = MathUtils.normVector(embedding);
		TopNList<Integer> topNlist = getTopN(embedding, topN);		
		Iterator<Integer> itObj = topNlist.getObjectList().iterator();
		Iterator<Double> itDist = topNlist.getObjectScore().iterator();
		while(itObj.hasNext()){
			Integer comWord = itObj.next();
			Double comDist = itDist.next();
			if(comDist.isNaN()){
				throw new RuntimeException("nan found during print topN");
			}
			out.println(words[comWord] + " -> " + comDist);
		}
	}
	
	public void printTopNEuclidean(double[] embedding, int topN, String[] words, PrintStream out){
		TopNList<Integer> topNlist = getTopNEuclidean(embedding, topN);		
		Iterator<Integer> itObj = topNlist.getObjectList().iterator();
		Iterator<Double> itDist = topNlist.getObjectScore().iterator();
		while(itObj.hasNext()){
			Integer comWord = itObj.next();
			Double comDist = itDist.next();
			if(comDist.isNaN()){
				throw new RuntimeException("nan found during print topN");
			}
			out.println(words[comWord] + " -> " + comDist);
		}
	}

	public void printTopNIfWord(String word, int topN, Vocab vocab, Vocab checkVocab, PrintStream out){
		printTopNIfWord(vocab.getEntry(word).getId(), topN, vocab, checkVocab, out);
	}

	public void printTopNIfWord(int word, int topN, Vocab vocab, Vocab checkVocab, PrintStream out){
		double[] vectorForWord = cosineDistA[word];
		printTopNIfWord(vectorForWord, topN, vocab, checkVocab, out);	
	}

	public void printTopNIfWord(double[] embedding, int topN, Vocab vocab, Vocab checkVocab, PrintStream out){
		TopNList<Integer> topNlist = getTopN(embedding, topN, vocab,checkVocab);		
		Iterator<Integer> itObj = topNlist.getObjectList().iterator();
		Iterator<Double> itDist = topNlist.getObjectScore().iterator();
		while(itObj.hasNext()){
			Integer comWord = itObj.next();
			Double comDist = itDist.next();
			String word = vocab.getEntryFromId(comWord).getWord();
			out.println(vocab.getEntryFromId(comWord).getWord() + "(" + vocab.getEntryFromId(comWord).count + ")" + " -> " + comDist);
		}
	}

	public void printTopNAnalogy(String[] from, String[] to, String woman, int topN, Vocab vocab){		
		double[] fromEmb = new double[vectors[0].length];
		double[] toEmb = new double[vectors[0].length];
		for(int v = 0; v < from.length; v++){
			double[] toVector = vectors[vocab.getEntry(to[v]).getId()];
			double[] fromVector = vectors[vocab.getEntry(from[v]).getId()];
			for(int i = 0; i < fromEmb.length; i++){
				toEmb[i] += (toVector[i])/from.length;
				fromEmb[i] += (fromVector[i])/from.length;
			}
		}

		double[] toVectorAvgNorm = MathUtils.normVector(toEmb);		
		double[] fromVectorAvgNorm = MathUtils.normVector(fromEmb);		
		double[] vectorForWord1 = cosineDistA[vocab.getEntry(woman).getId()];
		printTopNAnalogy(fromVectorAvgNorm, toVectorAvgNorm, vectorForWord1, topN, vocab);
	}

	public void printTopNAnalogy(String man, String king, String woman, int topN, Vocab vocab){
		double[] manEmb = cosineDistA[vocab.getEntry(man).getId()];
		double[] kingEmb = cosineDistA[vocab.getEntry(king).getId()];
		double[] womanEmb = cosineDistA[vocab.getEntry(woman).getId()];
		printTopNAnalogy(manEmb, kingEmb, womanEmb, topN, vocab);
	}

	public void printTopNAnalogy(double[] man, double[] king, double[] woman, int topN, Vocab vocab){
		TopNList<Integer> topNlist = getTopNAnalogy(man, king, woman, topN, vocab);		
		Iterator<Integer> itObj = topNlist.getObjectList().iterator();
		Iterator<Double> itDist = topNlist.getObjectScore().iterator();
		while(itObj.hasNext()){
			Integer comWord = itObj.next();
			Double comDist = itDist.next();
			System.err.println(vocab.getEntryFromId(comWord).getWord() + " -> " + comDist);
		}
	}

	public TopNList<Integer> getTopNAnalogy(double[] man, double[] king, double[] woman, int topN, Vocab vocab){
		TopNList<Integer> topNlist = new TopNList<Integer>(topN);
		for(int i = 0; i < cosineDistA.length; i++){
			double distanceManX = cosineSimilarity(man, cosineDistA[i]);
			double distanceKingX = cosineSimilarity(king, cosineDistA[i]);
			double distanceWomanX = cosineSimilarity(woman, cosineDistA[i]);
			double distance = distanceWomanX + distanceKingX - distanceManX;
			topNlist.add(i, distance);
		}
		return topNlist;
	}

	public TopNList<Integer> getTopN(double[] embedding, int topN){
		TopNList<Integer> topNlist = new TopNList<Integer>(topN);
		for(int i = 0; i < cosineDistA.length; i++){
			double distance = cosineSimilarity(embedding, cosineDistA[i]);
			topNlist.add(i, distance);
		}
		return topNlist;
	}

	public TopNList<Integer> getTopNEuclidean(double[] embedding, int topN){
		TopNList<Integer> topNlist = new TopNList<Integer>(topN);
		for(int i = 0; i < cosineDistA.length; i++){
			double distance = euclideanSimilarity(embedding, vectors[i]);
			topNlist.add(i, distance);
		}
		return topNlist;
	}
	
	public TopNList<Integer> getTopN(double[] embedding, int topN, Vocab inputVocab, Vocab checkVocab){
		TopNList<Integer> topNlist = new TopNList<Integer>(topN);
		for(int i = 0; i < cosineDistA.length; i++){
			String word = inputVocab.getEntryFromId(i).word;
			if(checkVocab.getEntry(word) != null){
				double distance = cosineSimilarity(embedding, cosineDistA[i]);
				topNlist.add(i, distance);
			}
		}
		return topNlist;
	}

	public static void kmeans(double[][] vectors, int classes, int[] wordToClass, double[][] centroids) {
		int totalIter = 10;
		int vocabSize = vectors.length;
		int dim = vectors[0].length;
		int[] centcn = new int[classes]; //count for each class
		for(int i = 0; i < vocabSize; i++){
			wordToClass[i] = i%classes; //assign classes randomly
		}
		for(int iter = 0; iter < totalIter; iter++){
			for(int c = 0; c < classes; c++){//initialize at 0
				for(int d = 0; d < dim; d++){
					centroids[c][d] = 0;
				}
				centcn[c] = 1;
			}
			for(int w = 0; w < vocabSize; w++){//compute centroid nominator
				for(int d = 0; d < dim; d++){
					if(wordToClass[w]==-1){
						System.err.println(w);						
					}
					centroids[wordToClass[w]][d] += vectors[w][d];
				}
				centcn[wordToClass[w]]++;
			}
			for(int c = 0; c < classes; c++){//normalize
				double closev = 0;
				for (int d = 0; d < dim; d++){
					centroids[c][d] /= centcn[c];
					closev += centroids[c][d] * centroids[c][d];
				}
				closev = Math.sqrt(closev);
				for (int d = 0; d < dim; d++){
					centroids[c][d] /= closev;
				}
			}
			for(int w = 0; w < vocabSize; w++){//reassign classes
				double closev = -1000;
				int closeid = -1;
				for(int c = 0; c < classes; c++){
					double x = 0;
					for(int d = 0; d < dim; d++){
						x+=centroids[c][d]*vectors[w][d];
					}
					if(x > closev){
						closev = x;
						closeid = c;
					}
				}
				wordToClass[w]=closeid;
			}
		}
	}

	public void kmeans(String file, int classes, Vocab vocab) {

		int vocabSize = vectors.length;
		int dim = vectors[0].length;
		int[] cl = new int[vocabSize]; // word to class
		double[][] cent = new double[classes][dim]; // mean for each class
		kmeans(vectors, classes, cl, cent);		

		//save classes
		try {
			PrintStream out = new PrintStream(new File(file));
			for(int w = 0; w < vocabSize; w++){
				out.println(cl[w] + " " + vocab.getEntryFromId(w).getWord());
			}
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(0);
		}
	}

	public void printAnalogy(int wordA, int wordB, int word1, int topN, Vocab vocab){
		double[] vectorForWordA = cosineDistA[wordA];
		double[] vectorForWordB = cosineDistA[wordB];
		double[] vectorForWord1 = cosineDistA[word1];
		double[] vectorForWord2 = new double[vectorForWord1.length];
		for(int i = 0; i < vectorForWord1.length; i++){
			vectorForWord2[i] = vectorForWord1[i] + (vectorForWordB[i] - vectorForWordA[i]);
		}
		printTopN(vectorForWord2, topN, vocab);
	}

	public void printAnalogyList(String[] from, String[] to, String word, int topN, Vocab vocab){
		double[] toVectorAvg = new double[vectors[0].length];
		double[] fromVectorAvg = new double[vectors[0].length];
		for(int v = 0; v < from.length; v++){
			double[] toVector = vectors[vocab.getEntry(to[v]).getId()];
			double[] fromVector = vectors[vocab.getEntry(from[v]).getId()];
			for(int i = 0; i < toVectorAvg.length; i++){
				toVectorAvg[i] += (toVector[i])/from.length;
				fromVectorAvg[i] += (fromVector[i])/from.length;
			}
		}

		double[] toVectorAvgNorm = MathUtils.normVector(toVectorAvg);		
		double[] fromVectorAvgNorm = MathUtils.normVector(fromVectorAvg);		
		double[] vectorForWord1 = cosineDistA[vocab.getEntry(word).getId()];
		double[] vectorForWord2 = new double[vectorForWord1.length];
		for(int i = 0; i < vectorForWord1.length; i++){
			vectorForWord2[i] = vectorForWord1[i] + toVectorAvgNorm[i] - fromVectorAvgNorm[i];
		}
		printTopN(vectorForWord2, topN, vocab);
	}

	public void printAverage(int wordA, int wordB, int topN, Vocab vocab){
		double[] vectorForWordA = cosineDistA[wordA];
		double[] vectorForWordB = cosineDistA[wordB];
		double[] vectorForWordC = new double[vectorForWordA.length];
		for(int i = 0; i < vectorForWordA.length; i++){
			vectorForWordC[i] = (vectorForWordB[i] + vectorForWordA[i]) / 2;
		}
		printTopN(vectorForWordC, topN, vocab);
	}

	public void printMinusAverage(int wordA, int wordB, int topN, Vocab vocab){
		double[] vectorForWordA = vectors[wordA];
		double[] vectorForWordB = vectors[wordB];
		double[] vectorForWordC = new double[vectorForWordA.length];
		for(int i = 0; i < vectorForWordA.length; i++){
			vectorForWordC[i] = (vectorForWordA[i] - vectorForWordB[i]);
		}
		printTopN(MathUtils.normVector(vectorForWordC), topN, vocab);
	}

	public void decompose(int word, int num, Vocab vocab){
		int numOfTopWords = 100;
		TopNList<Integer> topNlist = getTopN(vectors[word], numOfTopWords);
		double[][] topNEmbeddings = new double[numOfTopWords-1][vectors[0].length];
		Iterator<Integer> itObj = topNlist.getObjectList().iterator();
		int i = 0;
		while(itObj.hasNext()){
			Integer comWord = itObj.next();
			if(comWord == word) continue;
			topNEmbeddings[i++]=vectors[comWord];
		}
		int[] wordToClass = new int[numOfTopWords-1];
		double[][] centroids = new double[num][vectors[0].length];
		kmeans(topNEmbeddings, num, wordToClass, centroids);

		/*double[][] closestEmbeddingPerClass = new double[num][vectors[0].length];
		double[] bestDistancePerClass = new double[num];

		for(int w = 0; w < topNEmbeddings.length; w++){
			int cl = wordToClass[w];
			double sim = cosineSimilarity(cosineDistA[word], MathUtils.normVector(topNEmbeddings[w]));
			if(sim > bestDistancePerClass[cl]){
				bestDistancePerClass[cl] = sim;
				closestEmbeddingPerClass[cl] = topNEmbeddings[w];
			}
		}*/

		for(int c = 0; c < num; c++){
			System.err.println("centroid " + c);			
			printTopN(MathUtils.normVector(centroids[c]), 10, vocab);
		}
	}	

	public static void main(String[] args){
		
		String vocabFile = "/Users/lingwang/Documents/workspace/ContinuousVectors/data/vocab.txt";
		String vectorsfile = "/Users/lingwang/Documents/workspace/ContinuousVectors/data/vectors.bin";
		Vocab vocab = new Vocab();
		vocab.loadFromCountFile(vocabFile);
		WordVectors v = new WordVectors(vectorsfile);
		System.err.println(v.getTopSimilarFeatures(new double[]{0.01,0.2,0.11,0.01}, new double[]{-1,-0.22,3,2}, 4));
		v.printTopN(vocab.getEntry("beijing").getId(), 1000, vocab);

	}

	public double[] getVector(WordEntry entry) {		
		return vectors[entry.id];
	}
}
