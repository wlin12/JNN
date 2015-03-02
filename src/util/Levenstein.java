package util;

import java.util.LinkedList;

public class Levenstein {
	
	public static class EditOperation{
		int op; // 0 = insert, 1 = delete, 2 = substitution, 3 == correct
		String from;
		String to;				
		int i;
		int j;
		
		public EditOperation(int op, String from, String to, int i, int j) {
			super();
			this.op = op;
			this.from = from;
			this.to = to;
			this.i = i;
			this.j = j;
		}

		@Override
		public String toString() {
			String ret = "";
			if(op==0){
				ret+="insert " + to;
			}
			if(op==1){
				ret+="delete " + from;
			}
			if(op==2){
				ret+="replace " + from + " with " + to;
			}
			if(op==3){
				ret+="leave " + from;
			}
			return ret + " i = " + i + " j = " + j;
		}
		
		
	}
	
	public static LinkedList<EditOperation> minDistanceOperation(String word1, String word2){
		LinkedList<EditOperation> operations = new LinkedList<Levenstein.EditOperation>();
		int len1 = word1.length();
		int len2 = word2.length();
	 
		// len1+1, len2+1, because finally return dp[len1][len2]
		int[][] dp = new int[len1 + 1][len2 + 1];
	 
		for (int i = 0; i <= len1; i++) {
			dp[i][0] = i;
		}
	 
		for (int j = 0; j <= len2; j++) {
			dp[0][j] = j;
		}
	 
		//iterate though, and check last char
		for (int i = 0; i < len1; i++) {
			char c1 = word1.charAt(i);
			for (int j = 0; j < len2; j++) {
				char c2 = word2.charAt(j);
	 
				//if last two chars equal
				if (c1 == c2) {
					//update dp value for +1 length
					dp[i + 1][j + 1] = dp[i][j];
				} else {
					int replace = dp[i][j] + 1; // no replaces 
					int insert = dp[i][j + 1] + 1;
					int delete = dp[i + 1][j] + 1;
	 
					int min = insert;
					min = delete > min ? min : delete;
					dp[i + 1][j + 1] = min;
				}
			}
		}
		
		int min1 = len1;
		int min2 = len2;
		while(min1 != 0 && min2 != 0){
			int replace = Integer.MAX_VALUE;
			int insert = Integer.MAX_VALUE;
			int delete = Integer.MAX_VALUE;
			
			if(min1>0&&min2>0){
				replace = dp[min1-1][min2-1];
			}
			if(min2>0){
				insert = dp[min1][min2-1];
			}
			if(min1>0){
				delete = dp[min1-1][min2];
			}
			int min = replace > insert ? insert : replace;
			min = delete > min ? min : delete;
			if(min == replace){
				int op = 2;
				if(min == dp[min1][min2]) op = 3;
				min1--;
				min2--;
				operations.addFirst(new EditOperation(op, word1.substring(min1, min1+1), word2.substring(min2, min2+1), min1, min2));
			}
			else if(min == insert){

				min2--;
				operations.addFirst(new EditOperation(0, word1.substring(min1, min1+1), word2.substring(min2, min2+1), min1, min2));
			}
			else if(min == delete){
				min1--;
				operations.addFirst(new EditOperation(1, word1.substring(min1, min1+1), word2.substring(min2, min2+1), min1, min2));
			}
		}
		return operations;
	}
	
	public static int minDistance(String word1, String word2) {
		int len1 = word1.length();
		int len2 = word2.length();
	 
		// len1+1, len2+1, because finally return dp[len1][len2]
		int[][] dp = new int[len1 + 1][len2 + 1];
	 
		for (int i = 0; i <= len1; i++) {
			dp[i][0] = i;
		}
	 
		for (int j = 0; j <= len2; j++) {
			dp[0][j] = j;
		}
	 
		//iterate though, and check last char
		for (int i = 0; i < len1; i++) {
			char c1 = word1.charAt(i);
			for (int j = 0; j < len2; j++) {
				char c2 = word2.charAt(j);
	 
				//if last two chars equal
				if (c1 == c2) {
					//update dp value for +1 length
					dp[i + 1][j + 1] = dp[i][j];
				} else {
					int replace = dp[i][j] + 1;
					int insert = dp[i][j + 1] + 1;
					int delete = dp[i + 1][j] + 1;
	 
					int min = replace > insert ? insert : replace;
					min = delete > min ? min : delete;
					dp[i + 1][j + 1] = min;
				}
			}
		}
	 
		return dp[len1][len2];
	}
	
	public static int minDistance(String[] sent1, String[] sent2) {
		int len1 = sent1.length;
		int len2 = sent2.length;
	 
		// len1+1, len2+1, because finally return dp[len1][len2]
		int[][] dp = new int[len1 + 1][len2 + 1];
	 
		for (int i = 0; i <= len1; i++) {
			dp[i][0] = i;
		}
	 
		for (int j = 0; j <= len2; j++) {
			dp[0][j] = j;
		}
	 
		//iterate though, and check last char
		for (int i = 0; i < len1; i++) {
			String c1 = sent1[i];
			for (int j = 0; j < len2; j++) {
				String c2 = sent2[j];
	 
				//if last two chars equal
				if (c1.equals(c2)) {
					//update dp value for +1 length
					dp[i + 1][j + 1] = dp[i][j];
				} else {
					int replace = dp[i][j] + 1;
					int insert = dp[i][j + 1] + 1;
					int delete = dp[i + 1][j] + 1;
	 
					int min = replace > insert ? insert : replace;
					min = delete > min ? min : delete;
					dp[i + 1][j + 1] = min;
				}
			}
		}
	 
		return dp[len1][len2];
	}
	
	public static double getSimilarity(String a, String b){
		int max = Math.max(a.length(), b.length());
		return 1 - ((double) minDistance(a, b))/max;
	}
	
	public static double getSimilarity(String[] a, String[] b){
		int max = Math.max(a.length, b.length);
		return 1 - ((double) minDistance(a, b))/max;
	}
	
	public static void main(String[] args){
		System.err.println(getSimilarity("cat", "cats"));
		for(EditOperation op : minDistanceOperation("crzzzzzy","crazy")){
			System.err.println(op);
		}
	}
}
