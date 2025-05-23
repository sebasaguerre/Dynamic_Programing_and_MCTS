# Dynamic Programming for Infinite Horizon Problem and MCTS

Two programs:

- Implementation of DP backwards recursion value iteration algorithm for dynamic inventory management problem. Here we looked at the limiting distribution for the problem, followed by solving the average-cost Poisson equation and finally setting up and solving the Bellman equation.
- Implementation of the Monte Carlo Tree Search to play 4 Connect against a user via a terminal interface. Difficulty of MCTS enemy can be easily adjusted accordingly. During experiments conducted against a group of 30+ human candidates, the algorithm won every match.
    - MCTS difficulty settings (mcts_iter var):
        - Begginer: 100-500
        - Intermediate: 500-1000
        - Advanced: 1000-2500
        - Hard: 2500-5000
        - Expert: 5000-10000
        - Master: 10000+

  All algorithms were implemented from scratch
  



