import numpy as np
import random
import pickle
from checkers_env import checkers_env
import csv
import datetime
import os

# RESUME OPTION
RESUME = True
# path to the Q-table 
CHECKPOINT = "q_table_20250624-163727.pkl" 
OPPONENT_MODE = 'random' # 'learning' or 'random'

# q-learn-hyperparameters
ALPHA_START = 0.1
ALPHA_END = 0.01
GAMMA = 0.99
EPSILON_NEW = 0.2      # epsilon for a fresh training run
EPSILON_RESUME = 0.2  # epsilon when resuming 
EPISODES = 10000
# EPSILON BURST
EPSILON_RESET_FREQUENCY = 1500 
EPSILON_RESET_VALUE = 0.15   

def encode_state(board, player):
    # if player == 1 keep as is; if -1 flip vertically and change sign
    if player == 1:
        return board.tobytes()
    flipped = np.flipud(board) * -1
    return flipped.tobytes()

def get_all_legal_moves(env):
    # returns all legal moves for the current player
    moves = []
    must_capture, capture_moves = env.any_capture_possible(env.current_player)
    if must_capture:
        return list(capture_moves)
    for fr in range(env.size):
        for fc in range(env.size):
            if np.sign(env.board[fr, fc]) == env.current_player:
                piece = env.board[fr, fc]
                directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                max_dist = 1 if abs(piece) == 1 else env.size - 1
                for dr, dc in directions:
                    for dist in range(1, max_dist + 1):
                        tr, tc = fr + dr * dist, fc + dc * dist
                        if 0 <= tr < env.size and 0 <= tc < env.size:
                            if env.is_valid_move(fr, fc, tr, tc, False):
                                moves.append((fr, fc, tr, tc))
                        if abs(piece) == 1:
                            break
    return moves

def select_action(q_table, state, legal_moves, epsilon):
    # epsilon greedy
    if random.random() < epsilon:
        return random.choice(legal_moves)
    qs = [q_table.get((state, move), 0.0) for move in legal_moves]
    max_q = max(qs)
    best_moves = [move for move, q in zip(legal_moves, qs) if q == max_q]
    return random.choice(best_moves)

def random_opponent_move(env):
    # opponent plays randomly
    moves = get_all_legal_moves(env)
    return random.choice(moves) if moves else None

def main():
    # main 
    env = checkers_env()
    stats = []

    # load or create the Q-table
    if RESUME and os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "rb") as f:
            saved_data = pickle.load(f)
            if OPPONENT_MODE == 'learning' and isinstance(saved_data, tuple):
                 q_table, opp_q = saved_data
            else:
                 q_table = saved_data
                 opp_q = {} # Create empty opp_q if not in checkpoint or not needed
        print(f"Resuming training from {CHECKPOINT}.")
        epsilon = EPSILON_RESUME
    else:
        q_table = {}
        opp_q = {}
        epsilon = EPSILON_NEW
        print("Starting a new training session.")

    # unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"training_log_{timestamp}.csv"
    q_table_file = f"q_table_{timestamp}.pkl"

    epsilon_final = 0.05 # The target minimum epsilon
    with open(log_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["episode", "total_reward", "steps", "outcome", "illegal_moves", "epsilon"])
        for ep in range(EPISODES):
            env.reset()
            done = False
            total_reward = 0
            steps = 0
            outcome = "draw"
            illegal_moves = 0
            while not done:
                # --- agent turn ---
                state = encode_state(env.board, env.current_player)
                legal_moves = get_all_legal_moves(env)
                if not legal_moves:
                    outcome = "loss"
                    break
                action = select_action(q_table, state, legal_moves, epsilon)
                board, reward, done, _ = env.step(action)

                if reward == -5: # check for illegal move penalty
                    illegal_moves += 1
                
                # opponent's turn
                if OPPONENT_MODE == 'learning':
                    opp_state = encode_state(env.board, env.current_player)
                    opp_reward = 0
                    if not done:
                        opp_moves = get_all_legal_moves(env)
                        if opp_moves:
                            if random.random() < 0.10: # 10% exploration for opponent
                                opp_action = random.choice(opp_moves)
                            else:
                                opp_qs = [opp_q.get((opp_state, m), 0.0) for m in opp_moves]
                                max_opp_q = max(opp_qs)
                                opp_action = random.choice([m for m, q in zip(opp_moves, opp_qs) if q == max_opp_q])
                            
                            _, opp_reward, done, _ = env.step(opp_action)
                            if opp_reward < 0:
                                opp_reward = 0
                        else:
                            done = True
                            reward += 10
                else: # random opponent
                    opp_reward = 0
                    if not done:
                        opp_move = random_opponent_move(env)
                        if opp_move:
                            _, opp_reward, done, _ = env.step(opp_move)
                        else:
                            done = True
                            reward += 10

                total_r = reward - opp_reward
                next_state = encode_state(env.board, env.current_player)

                # Q-update for main agent
                alpha = ALPHA_START - (ALPHA_START - ALPHA_END) * min(1.0, (ep+1)/EPISODES)
                next_legal = get_all_legal_moves(env) if not done else []
                old_q = q_table.get((state, action), 0.0)
                target = total_r
                if next_legal:
                    target += GAMMA * max(q_table.get((next_state, a), 0.0) for a in next_legal)
                q_table[(state, action)] = old_q + alpha * (target - old_q)

                # Q-update for opponent agent
                if 'opp_action' in locals():
                    next_opp_state = encode_state(env.board, env.current_player)
                    next_opp_legal = get_all_legal_moves(env) if not done else []
                    old_opp_q = opp_q.get((opp_state, opp_action), 0.0)
                    opp_target = -total_r # opponent's reward is the negative of the agent's
                    if next_opp_legal:
                        # 'max' here is from the perspective of the next player, which is our main agent.
                        opp_target += GAMMA * max(q_table.get((next_opp_state, a), 0.0) for a in next_opp_legal)
                    opp_q[(opp_state, opp_action)] = old_opp_q + alpha * (opp_target - old_opp_q)

                total_reward += total_r
                steps += 1

                if done:
                    if total_r > 0:
                        outcome = "win"
                    elif total_r < 0:
                        outcome = "loss"
                    break
                if steps > 200:
                    outcome = "timeout"
                    break
            stats.append(total_reward)
            writer.writerow([ep+1, total_reward, steps, outcome, illegal_moves, epsilon])
            
            # Exponential epsilon decay
            epsilon *= 0.9995
            epsilon = max(epsilon, epsilon_final)

            # Epsilon burst/reset
            if (ep + 1) % EPSILON_RESET_FREQUENCY == 0:
                epsilon = EPSILON_RESET_VALUE
                print(f"*** EPSILON BURST: Reset to {epsilon} at episode {ep+1} ***")

            if (ep+1) % 100 == 0:
                avg = np.mean(stats[-100:])
                print(f"Episode {ep+1}/{EPISODES}, avg reward (last 100): {avg:.2f}, steps: {steps}, last outcome: {outcome}, epsilon: {epsilon:.3f}")
    # save final Q-tables
    with open(q_table_file, "wb") as f:
        if OPPONENT_MODE == 'learning':
            pickle.dump((q_table, opp_q), f)
        else:
            pickle.dump(q_table, f)
    print(f"Training complete. Q-tables saved to {q_table_file}. Log saved to {log_file}")

if __name__ == "__main__":
    main() 