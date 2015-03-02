package vocab;

import java.io.PrintStream;
import java.util.Scanner;

import util.SerializeUtils;

public class WordEntry {
	
	public String word;	
	public int count = 0;
	public int[] code;
	public int[] point;
	public int id;
	
	
	public WordEntry(String word) {
		this.word = word;
	}
	
	@Override
	public String toString() {
		return word + "(count:" + count + ", code:" + getCodeStr() + ", point:" + getPointStr() + ")";
	}

	private String getPointStr() {
		String s = "";
		for(int c : point){
			s+=c + " ";
		}
		return s;
	}

	private String getCodeStr() {
		String s = "";
		for(int c : code){
			s+=c;
		}
		return s;
	}
	
	public int getId() {
		return id;
	}

	public int[] getCode() {
		return code;
	}
	
	public int[] getPoint() {
		return point;
	}
	
	public String getWord() {
		return word;
	}
	
	public int getCount() {
		return count;
	}

	public void save(PrintStream out) {
		out.println(id);
		out.println(word);
		out.println(count);
		SerializeUtils.saveIntArray(code, out);
		SerializeUtils.saveIntArray(point, out);
	}
	
	public void load(Scanner in){
		id = Integer.parseInt(in.nextLine());
		word = in.nextLine();
		count = Integer.parseInt(in.nextLine());
		code = SerializeUtils.loadIntArray(in);
		point = SerializeUtils.loadIntArray(in);
		in.nextLine();
	}

	public void addCount(int i) {
		count+=i;
	}
}
