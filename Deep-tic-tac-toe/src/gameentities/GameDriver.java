package gameentities;

import java.util.Arrays;
import java.util.List;

import py4j.GatewayServer;

public class GameDriver {
	
	private int[] board = new int[9];
	private boolean isDone = true;
	
	public GameDriver() {
	}
	

	public static void main(String[] args) {
		GameDriver app = new GameDriver();
		 // app is now the gateway.entry_point
		GatewayServer server = new GatewayServer(app);
		server.start();	
	}
	
	public int innitGame() {
		board = new int[9];
		// Swap between 1 and 2
		isDone = false;
		return 1;
	}
	
	public int[] getState() {
		return board;
	}
	
	public boolean isDone() {
		return isDone;
	}
	
	public int step(List<Integer> playerAndSpace) {
		if(isDone) {
			return -5;
		}
		int player = playerAndSpace.get(0);
		int space = playerAndSpace.get(1);
		
		if (board[space] != 0) {
			return -1;
		}
		board[space] = player;
		
		if(checkWinner(player)) {
			isDone = true;
			return 5;
		}
		return 0;
	}
	
	
	private boolean checkWinner(int valuePlayed) {
		boolean winner = false;
		
		if (board[0] == valuePlayed) {
			winner = winner|validateSet(Arrays.asList(0, 1, 2));
			winner = winner|validateSet(Arrays.asList(0, 3, 6));
			winner = winner|validateSet(Arrays.asList(0, 4, 8));
		}
		
		if (board[6] == valuePlayed) {
			winner = winner|validateSet(Arrays.asList(6, 4, 2));
			winner = winner|validateSet(Arrays.asList(6, 7, 8));
		}
		
		if (board[4] == valuePlayed) {
			winner = winner|validateSet(Arrays.asList(3, 4, 5));
			winner = winner|validateSet(Arrays.asList(2, 4, 7));
		}
		if (board[5] == valuePlayed) {
			winner = winner|validateSet(Arrays.asList(2, 5, 8));
		}
		return winner;
	}
	
	private boolean validateSet(List<Integer> places) {
		if (board[places.get(0)] == board[places.get(1)] && board[places.get(0)] == board[places.get(2)]) {
			return true;
		}
		
		return false;
	}
	
}
