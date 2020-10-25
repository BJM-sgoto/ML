mcts1: 
 - complete copy from https://github.com/junxiaosong/AlphaZero_Gomoku
mcts2: 
 - mcts1 with some minor configs
mcts3: 
 - fix function <get_equi_data> in main.py to flip and rotate data
 - change board class, not to revest board vertically
mcts4: 
 - change shape of board to : height X width X 1
mcts5: 
 - change the entire code of board class
mcts6:
 - change graphic code
 - change neural_network structure, only generate probablity of valid moves
mcts7: 
 - run on 8 X 8 board
mcts9:
 - implement "parallel" mcts
mctsa:
 - improve "parallel" mcts by 
	- enable board export and import state dirrectly
	- pass multiple states to GPU
mctsb: (incorrect) => remove later
 - compute value of node once when updating
 - change n_visits once
 - not use undo_move
mctsc:
 - let 2 models compete
 
 
mcts8: => run again
 - improve performance using numpy array ( X )
 - test against pure mcts
mctsd: 
 - let 2 models compete
 - combine with improvement from mcts8
mctse: 
 - drop too much data => rewrite update_policy function
  - run multiple times
	- drop entropy
	- no need to change learning rate

TODO:
fix 8
remove b
fix c
fix d
rename c->b
rename d->c
