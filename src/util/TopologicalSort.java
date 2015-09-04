package util;

import java.util.*;

import org.apache.commons.math3.util.FastMath;

// obtained from https://sites.google.com/site/indy256/algo/topological_sorting
// then added a lot of stuff
public class TopologicalSort {

	static void dfs(List<Integer>[] graph, boolean[] used, List<Integer> res, int u) {
		used[u] = true;
		for (int v : graph[u]){
			if (!used[v]){
				dfs(graph, used, res, v);
			}
		}
		res.add(u);
	}
	
	static void dfsIterative(List<Integer>[] graph, boolean[] used, LinkedList<Integer> res, int u){
		LinkedList<Integer> stack = new LinkedList<Integer>();
		LinkedList<Integer> toExpand = new LinkedList<Integer>();
		toExpand.add(u);
		while(!toExpand.isEmpty()){
			int v = toExpand.removeFirst();
			if(used[v]){
				continue;
			}			
			stack.add(v);
			used[v] = true;
			for(int k : graph[v]){
				if(!used[k])
				{
					toExpand.addFirst(k);
				}
			}
		}
		Iterator<Integer> it = stack.descendingIterator();
		while(it.hasNext()){
			res.add(it.next());
		}
	}

	public static List<Integer> topologicalSort(List<Integer>[] graph) {
		int n = graph.length;
		boolean[] used = new boolean[n];
		LinkedList<Integer> res = new LinkedList<>();
		used = new boolean[n];
		for (int i = 0; i < n; i++){
			if (!used[i])
				dfs(graph, used, res, i);
		}
				
		Collections.reverse(res);
		return res;
	}
	
	public static List<Integer> topologicalSortBFS(List<Integer>[] graph) {
		HashSet<Integer>[] incoming = new HashSet[graph.length];
		int edges = 0;
		HashSet<Integer> nodes = new HashSet<Integer>();
		for(int i = 0; i < graph.length; i++){
			incoming[i] = new HashSet<Integer>();
			nodes.add(i);
		}

		//build reverse graph
		for(int i = 0; i < graph.length; i++){
			for(int v : graph[i]){
				incoming[v].add(i);
				edges++;
			}
		}
		
		List<Integer> ret = new LinkedList<Integer>();
		
		LinkedList<Integer> S = new LinkedList<Integer>();
		for(int n : nodes){
			if(incoming[n].size() == 0){
				S.add(n);
			}
		}
		
		while(S.size()>0){
			int n = S.removeFirst();
			nodes.remove(n);
			for(int v : graph[n]){
				incoming[v].remove(n);
				edges--;
				if(incoming[v].size()==0){
					S.add(v);
				}
			}
			ret.add(n);
		}
		if(edges > 0){
			throw new RuntimeException("found cycles in the graph");
		}
		return ret;
	}

	
	public static List<List<Integer>> blockTopologicalSort(List<Integer>[] graph){
		List<Integer> topologicalSort = topologicalSortBFS(graph);
		List<List<Integer>> ret = new LinkedList<List<Integer>>();
		
		int[] nodeToTimetamp = new int[topologicalSort.size()];
		
		int timestamp = 0;
		for(int node : topologicalSort){
			nodeToTimetamp[node] = timestamp++;
		}
			
		LinkedList<Integer> block = new LinkedList<Integer>();
		int minTimestamp = topologicalSort.size();
		timestamp = 0;
		for(int node : topologicalSort){
			if(timestamp == minTimestamp){
				ret.add(block);
				block = new LinkedList<Integer>();
				minTimestamp = topologicalSort.size();
			}
			block.add(node);
			for(int next : graph[node]){
				minTimestamp = FastMath.min(minTimestamp, nodeToTimetamp[next]);
			}
			timestamp++;
		}
		if(block.size() >0){
			ret.add(block);
		}
		return ret;
	}
	
	public static int[] intervalGroups(int[] starts, int[] ends){
		ArrayList<Integer> boundaries = new ArrayList<Integer>();
		int[] startBlocks = new int[starts.length];
		int[] endBlocks = new int[ends.length];
		int i = 0;
		int j = 0;
		
		while(true){
			int startPos = Integer.MAX_VALUE;
			int endPos = Integer.MAX_VALUE;
			if(i<starts.length){
				startPos = starts[i];
			}
			if(j<ends.length){
				endPos = ends[j];
			}
			if(startPos>endPos){
				endBlocks[j] = boundaries.size();
				boundaries.add(endPos);
				j++;
			}
			else if(startPos<endPos){
				startBlocks[i] = boundaries.size();
				boundaries.add(startPos);
				i++;
			}
			else{
				startBlocks[i] = boundaries.size();
				endBlocks[j] = boundaries.size();
				boundaries.add(startPos);
				i++;
				j++;
			}
			if(i >= starts.length && j>=ends.length){
				break;
			}
		}
		
		int[] segmentScores = new int[boundaries.size()-1];		
		for(int k = 0; k < startBlocks.length; k++){
			for(int pos = startBlocks[k]; pos <= endBlocks[k]; pos++){
				segmentScores[pos]++;
			}
		}
		return null;//to do
	}

	// Usage example
	public static void main(String[] args) {
		int n = 6;
		List<Integer>[] g = new List[n];
		for (int i = 0; i < n; i++) {
			g[i] = new ArrayList<>();
		}
		g[0].add(3);
		g[0].add(1);
		g[1].add(2);
		g[3].add(1);
		g[3].add(2);
		g[3].add(4);
		g[5].add(0);

		System.out.println(topologicalSortBFS(g));
		//System.out.println(blockTopologicalSort(g));
		
		
	}
}