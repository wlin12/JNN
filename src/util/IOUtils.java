package util;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.StringReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Stack;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class IOUtils {
	static BufferedReader in = new BufferedReader(new InputStreamReader(System.in));

	public static BufferedReader getIn(){
		return in;
	}

	public static String getLineFromInput(){
		String s = "";
		while (s.length() == 0){
			try {
				s = getIn().readLine();
			} catch (IOException e) {
				e.printStackTrace();
				System.exit(0);
				return null;
			} 
		}
		return s;
	}

	public static DataOutputStream getDataOutputStream(String file){
		String folder = new File(file).getParent();
		if(!folder.equals("") && !new File(folder).exists()){
			mkdir(folder);
		}
		DataOutputStream out;
		try {
			if(file.endsWith(".gz")){				
				out = new DataOutputStream(new GZIPOutputStream(new FileOutputStream(file)));
			}
			else {
				out = new DataOutputStream(new FileOutputStream(file));
			}
			return out;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(0);
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
			System.exit(0);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}		
		return null;
	}
	
	public static PrintStream getPrintStream(String file){
		return getPrintStream(file, false);
	}

	public static PrintStream getPrintStream(String file, boolean autoflush){
		String folder = new File(file).getParent();
		if(!folder.equals("") && !new File(folder).exists()){
			mkdir(folder);
		}
		PrintStream out;
		try {
			if(file.endsWith(".gz")){				
				out = new PrintStream(new GZIPOutputStream(new FileOutputStream(file)), autoflush, "UTF-8");
			}
			else {
				out = new PrintStream(new FileOutputStream(file), autoflush, "UTF-8");
			}
			return out;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(0);
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
			System.exit(0);
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}		
		return null;
	}

	public static PrintWriter getPrintWriter(String file){
		PrintWriter out;
		try {
			out = new PrintWriter(new FileOutputStream(file), false);
			return out;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.exit(0);
		}
		return null;
	}
	
	public static DataInputStream getDataInputStream(String file){
		try {
			if(file.endsWith(".gz")){				
				return new DataInputStream(
						new GZIPInputStream(new FileInputStream(file)));
			}
			return new DataInputStream(new FileInputStream(file));
		} catch (UnsupportedEncodingException e) {
			System.err.println("error reading " + file);
			e.printStackTrace();
			System.exit(0);
		} catch (FileNotFoundException e) {
			System.err.println("error reading " + file);
			e.printStackTrace();
			System.exit(0);
		} catch (IOException e) {
			System.err.println("error reading " + file);
			e.printStackTrace();
			System.exit(0);
		}		
		return null;
	}

	public static BufferedReader getReader(String file){
		try {
			if(file.endsWith(".gz")){				
				return new BufferedReader( new InputStreamReader(
						new GZIPInputStream(new FileInputStream(file)), "UTF-8"));
			}
			return new BufferedReader( new InputStreamReader(
					new FileInputStream(file), "UTF-8"));
		} catch (UnsupportedEncodingException e) {
			System.err.println("error reading " + file);
			e.printStackTrace();
			System.exit(0);
		} catch (FileNotFoundException e) {
			System.err.println("error reading " + file);
			e.printStackTrace();
			System.exit(0);
		} catch (IOException e) {
			System.err.println("error reading " + file);
			e.printStackTrace();
			System.exit(0);
		}		
		return null;
	}

	public static BufferedReader getReaderWithExceptions(String file) throws IOException{
		try {
			if(file.endsWith(".gz")){				
				return new BufferedReader( new InputStreamReader(
						new GZIPInputStream(new FileInputStream(file)), "UTF-8"));
			}
			return new BufferedReader( new InputStreamReader(
					new FileInputStream(file), "UTF-8"));
		} catch (UnsupportedEncodingException e) {
			System.err.println("error reading " + file);
			e.printStackTrace();
			System.exit(0);
		} catch (FileNotFoundException e) {
			System.err.println("error reading " + file);
			e.printStackTrace();
			System.exit(0);
		} catch (IOException e) {
			System.err.println("error reading " + file);
			e.printStackTrace();
			throw e;
		}		
		return null;
	}

	public static boolean exists(String oldFile) {
		return new File(oldFile).exists() || new File(oldFile+".gz").exists();
	}

	public static boolean equal(String newFile, String oldFile, int linesToCompare) {
		BufferedReader newFileReader = getReader(newFile);
		BufferedReader oldFileReader = getReader(oldFile);
		try {
			int line = 0;
			while(newFileReader.ready() && oldFileReader.ready() && line++ < linesToCompare){
				if(!newFileReader.readLine().equals(oldFileReader.readLine())){
					return false;
				}
			}
			int oldLine = 0;
			int newLine = 0;
			while(newFileReader.ready()){
				newLine++;
				newFileReader.readLine();
			}
			while(oldFileReader.ready()){
				oldLine++;
				oldFileReader.readLine();
			}
			if(oldLine != newLine){
				return false;
			}
			newFileReader.close();
			oldFileReader.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
			return false;
		}
		return true;
	}

	public static void copyfile(String srFile, String dtFile){
		try{
			File f1 = new File(srFile);
			File f2 = new File(dtFile);
			InputStream in = new FileInputStream(f1);

			//For Append the file.
			//  OutputStream out = new FileOutputStream(f2,true);

			//For Overwrite the file.
			OutputStream out = null;
			if(dtFile.endsWith(".gz")){
				out = new GZIPOutputStream(new FileOutputStream(f2));
			}
			else{
				out = new FileOutputStream(f2);				
			}

			byte[] buf = new byte[1024];
			int len;
			while ((len = in.read(buf)) > 0){
				out.write(buf, 0, len);
			}
			in.close();
			out.close();
		}
		catch(FileNotFoundException ex){
			System.err.println(ex.getMessage() + " in the specified directory.");
			System.exit(0);
		}
		catch(IOException e){
			System.err.println(e.getMessage());  
			System.exit(0);

		}
	}

	public static void deleteFile(String newFile) {
		new File(newFile).delete();
	}

	public static void replaceFile(String fileToReplace, String fileToReplaceWith){
		copyfile(fileToReplaceWith, fileToReplace);
		deleteFile(fileToReplaceWith);
	}

	public static void addLinesToFile(String[] newLines, String file){
		ArrayList<String> lines = new ArrayList<String>();
		if(new File(file).exists()){
			try {
				BufferedReader reader = IOUtils.getReader(file);
				while(reader.ready()){
					String line = reader.readLine();
					if(!line.equals("")){
						lines.add(line);
					}
				}
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
				System.exit(0);
			}
		}
		for(String newLine : newLines){
			if(!lines.contains(newLine)){
				lines.add(newLine);
			}
			else{
				//System.err.print("skipping line " + newLine);
			}
		}
		PrintStream out = IOUtils.getPrintStream(file);
		for(String user : lines){
			out.println(user);
		}
		out.close();
	}

	public static void concatFiles(String[] files, String output){
		PrintStream out = IOUtils.getPrintStream(output);
		for(String file : files){
			BufferedReader reader = IOUtils.getReader(file);
			try {
				while(reader.ready()){
					out.println(reader.readLine());
				}
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
				System.exit(0);
			}
		}
	}

	public static String[] list(String dir) {
		String[] list = new File(dir).list(new FilenameFilter() {

			@Override
			public boolean accept(File dir, String name) {
				if (!name.endsWith("~") && !name.startsWith(".")) {
					return true;
				}
				return false;
			}
		});
		return list;
	}

	public static String[] listRecursive(String dir) {		
		final Stack<String> dirs = new Stack<String>();
		if(!dir.endsWith("/")){
			dir+="/";
		}
		dirs.push(dir);
		ArrayList<String> files = new ArrayList<String>();
		while(!dirs.isEmpty()){
			String currDir = dirs.pop();
			String[] list = new File(currDir).list(new FilenameFilter() {

				@Override
				public boolean accept(File dir, String name) {
					if (!name.endsWith("~") && !name.startsWith(".")) {
						return true;
					}
					return false;
				}
			});
			String pred = currDir.replaceFirst(dir, "");
			for(String file : list){
				String fullpath = currDir + "/" + file;
				if(new File(fullpath).isDirectory()){
					dirs.push(fullpath);
				}
				else{
					files.add((pred + "/" + file).replaceFirst("/", ""));
				}
			}			
		}
		return files.toArray(new String[0]);
	}

	public static BufferedReader getBufferedReaderFromStringArray(String[] list){
		StringBuilder buffer = new StringBuilder();
		for(int i = 0; i < list.length; i++) {
			String current = list[i];			
			buffer.append(current);
			if(i != (list.length - 1)){
				buffer.append("\n");
			}		    
		}
		BufferedReader br = new BufferedReader(new StringReader(buffer.toString()));
		return br;
	}

	public static void printToFile(String file, String scriptToBuild) {		
		PrintStream out = getPrintStream(file);
		out.println(scriptToBuild);
		out.close();
	}

	public static void mkdir(String evalDir) {
		new File(evalDir).mkdirs();
	}

	public static void printDirToFile(String dir,
			PrintStream out) {
		String[] files = IOUtils.listRecursive(dir);
		for(String file : files){
			BufferedReader reader = IOUtils.getReader(dir + "/" + file);
			try {
				while(reader.ready()){
					String line = reader.readLine();
					out.println(line);
				}
				reader.close();
			} catch (IOException e) {
				e.printStackTrace();
				System.exit(0);
			}
		}
		out.close();
	}

	public static String readLineFromFile(String inputFile){
		BufferedReader reader = IOUtils.getReader(inputFile);		
		try {
			String line = reader.readLine();
			reader.close();
			return line;
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public static String[] readFileLines(String inputFile){
		BufferedReader reader = IOUtils.getReader(inputFile);		
		LinkedList<String> lines = new LinkedList<String>();
		try {
			while(reader.ready()){
				String line = reader.readLine();
				lines.addLast(line);
			}
			reader.close();
			return lines.toArray(new String[0]);
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}			

	}

	public static void main(String[] args){
		String[] list = listRecursive("/Users/lingwang/Documents/servers/server/translationLexicons");
		for(String l : list){
			System.err.println(l);
			System.err.println(new File("/Users/lingwang/Documents/servers/server/translationLexicons/"+l).lastModified());
		}
	}

	public static void compressFile(String originalFile) {
		String output = originalFile + ".gz";
		try {
			PrintStream out = new PrintStream(new GZIPOutputStream(new FileOutputStream(output)), false, "UTF-8");
			BufferedReader reader = getReader(originalFile);
			while(reader.ready()){
				out.println(reader.readLine());
			}
			out.close();
			reader.close();
			IOUtils.deleteFile(originalFile);
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static int getNumberOfLines(String output) {
		BufferedReader reader = IOUtils.getReader(output);
		int numberOfLines = 0;
		try {
			while (reader.ready()){
				reader.readLine();
				numberOfLines++;
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
			return 0;
		}
		return numberOfLines;
	}
	
	public static long getNumberOfLinesLong(String output) {
		BufferedReader reader = IOUtils.getReader(output);
		long numberOfLines = 0;
		try {
			while (reader.ready()){
				reader.readLine();
				numberOfLines++;
			}
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
			return 0;
		}
		return numberOfLines;
	}

	public static void writeCommandToFile(String command,
			String file) {
		try {
			PrintStream out = getPrintStream(file);
			Process p = Runtime.getRuntime().exec(command);
			BufferedReader stdInput = new BufferedReader(new 
					InputStreamReader(p.getInputStream()));
			String s = null;
			while ((s = stdInput.readLine()) != null) {
				out.println(s);
			}
			stdInput.close();
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(0);
		}
	}

	public static InputStream getInputStream(String file){
		try {

			if(file.endsWith(".gz")){				
				return new GZIPInputStream(new FileInputStream(file));
			}
			return new FileInputStream(file);
		} catch (IOException e) {
			System.err.println("error reading " + file);
			e.printStackTrace();
			System.exit(0);
		}
		return null;

	}
	
	public static double[] lineIntoDoubleArray(String line, String delimiter){
		String[] doubleStr = line.split(delimiter);
		double[] doubles = new double[doubleStr.length];
		for(int i = 0; i < doubles.length; i++){
			doubles[i] = Double.parseDouble(doubleStr[i]);
		}
		return doubles;
	}

	public static String doubleToString(double[] hidden) {
		String ret = "";
		for(int i = 0; i< hidden.length ; i++){
			if(i != 0){
				ret+=" ";
			}
			ret += hidden[i];
		}		
		return ret;
	}
	
	public static String doubleToStringWithPos(double[] hidden) {
		String ret = "";
		for(int i = 0; i< hidden.length ; i++){
			if(i != 0){
				ret+=" ";
			}
			ret += hidden[i] +"(" + i + ")";
		}		
		return ret;
	}
	
	public static class iterateFilesCallback{
		public void cb(String[] lines, int lineNumber){}
	}
	
	public static void iterateFiles(String[] files, iterateFilesCallback cb){
		BufferedReader[] readers = new BufferedReader[files.length];
		for(int i = 0; i < readers.length; i++){
			readers[i] = getReader(files[i]);
		}
		String line = null;
		String[] lines = new String[files.length];
		try {
			int lineNumber = 0;
			while((line = readers[0].readLine()) != null){
				lines[0] = line;
				for(int i = 1; i < readers.length; i++){
					lines[i] = readers[i].readLine();
				}
				cb.cb(lines, lineNumber++);
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}
	
	public static void iterateFiles(String file, iterateFilesCallback cb){
		iterateFiles(new String[]{file}, cb);
	}
}
