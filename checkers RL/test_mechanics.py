import numpy as np
from checkers_env import checkers_env

def make_empty_env():
    env = checkers_env()
    env.board[:] = 0          # clear the board
    env.current_player = 1
    env.done = False
    return env

def run_all_tests():
    failures = []

    def check(name, cond):
        if cond:
            print(f"{name:40} PASS")
        else:
            print(f"{name:40} FAIL")
            failures.append(name)

    # 1 simple pawn forward move
    env = make_empty_env()
    env.board[5,0] = 1
    env.current_player = 1
    s, r, done, _ = env.step((5,0,4,1))
    check("pawn forward move",
          r == 0 and s[4,1] == 1 and s[5,0] == 0 and not done)

    # 2 mandatory-capture penalty (quiet move when capture available)
    env = make_empty_env()
    env.board[5,2] = 1
    env.board[4,3] = -1
    env.current_player = 1
    s, r, done, _ = env.step((5,2,4,1))  # quiet move
    check("mandatory-capture penalty",
          r == -1 and s[5,2] == 0 and not done)

    # 3 pawn capture reward
    env = make_empty_env()
    env.board[5,2] = 1
    env.board[4,3] = -1
    env.current_player = 1
    s, r, done, _ = env.step((5,2,3,4))  # jump
    check("pawn capture reward",
          r == 12 and s[3,4] == 1 and s[4,3] == 0 and done)

    # 4 backward pawn capture
    env = make_empty_env()
    env.board[3,4] = 1
    env.board[2,3] = -1
    env.current_player = 1
    s, r, done, _ = env.step((3,4,1,2))
    check("backward pawn capture",
          r == 12 and s[1,2] == 1 and done)

    # 5 promotion
    env = make_empty_env()
    env.board[1,2] = 1
    env.current_player = 1
    s, r, done, _ = env.step((1,2,0,1))
    check("pawn promotion",
          r == 5 and s[0,1] == 2)

    # 6 queen free move any distance
    env = make_empty_env()
    env.board[2,3] = 2
    env.current_player = 1
    s, r, done, _ = env.step((2,3,0,1))
    check("queen free move",
          r == 0 and s[0,1] == 2)

    # 7 queen short capture
    env = make_empty_env()
    env.board[5,2] = 2
    env.board[4,3] = -1
    env.current_player = 1
    s, r, done, _ = env.step((5,2,3,4))
    check("queen short capture",
          r == 12 and s[3,4] == 2 and s[4,3] == 0 and done)

    # 8 queen long capture
    env = make_empty_env()
    env.board[5,2] = 2
    env.board[4,3] = -1
    env.current_player = 1
    s, r, done, _ = env.step((5,2,1,6))
    check("queen long capture",
          r == 12 and s[1,6] == 2 and s[4,3] == 0 and done)

    # 9 multi-capture sequence (winning move)
    env = make_empty_env()
    env.board[5,2] = 1
    env.board[4,3] = -1
    env.board[2,5] = -1
    env.current_player = 1
    moves = [(5,2,3,4), (3,4,1,6)]
    s, r, done, _ = env.step(moves)
    check("multi-capture sequence",
          r == 14 and s[1,6] == 1 and s[4,3] == 0 and s[2,5] == 0 and done)

    # 10 illegal move penalty (play continues)
    env = make_empty_env()
    env.board[5,2] = 1
    env.current_player = 1
    s, r, done, _ = env.step((5,2,5,4))  # horizontal move
    check("illegal move penalty",
          r == -3 and not done and s[5,2] == 0)

    # 11 win by capturing last piece
    env = make_empty_env()
    env.board[5,2] = 1       # your pawn
    env.board[4,3] = -1      # single enemy pawn
    env.current_player = 1
    # jump over (4,3) to (3,4), capturing the last enemy
    s, r, done, _ = env.step((5,2,3,4))
    check("win reward",
          r == 12 and done)

    # 12 incomplete multi-capture penalty
    env = make_empty_env()
    env.board[5,2] = 2
    env.board[4,3] = -1
    env.board[2,5] = -1
    env.current_player = 1
    s, r, done, _ = env.step([(5,2,3,4)])  # only one of two possible jumps
    check("incomplete multi-capture penalty",
          r == -1 and s[3,4] == 0 and not done)

    # summary
    if failures:
        print(f"\n{len(failures)} test(s) failed: {failures}")
        exit(1)
    else:
        print("\nall tests passed!")
        exit(0)

if __name__ == "__main__":
    run_all_tests()