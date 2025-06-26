"""
Checkers environment for RL

Board: 8x8 numpy array
  0: empty
  1: player 1 pawn
  2: player 1 queen
 -1: player 2 pawn
 -2: player 2 queen

Rules:
- pawns move diagonally forward, one step at a time
- pawns capture diagonally in any direction (including backwards) by jumping over opponent's piece
- queens move and capture diagonally any distance in any direction
- multi-capture is allowed in a single turn
- if a capture is possible, it must be taken (mandatory capture)
- if a player does not capture when able, the piece that failed to capture is removed and a -2 penalty is applied
- if a player makes an illegal move, the offending piece is removed, a -5 penalty is applied, but play continues
- promotion: pawn becomes queen when reaching the last row
- rewards: +2 per capture, +5 for promotion, -3 for illegal move, -1 for failing to capture when able, +10 for winning the game, -2 for failing to capture when able or failing to continue a multi-capture
- game ends when you eat all opponent's pieces

move format: (fr, fc, tr, tc) where fr/fc is from row/col, tr/tc is to row/col
multi-capture: pass a list of such tuples
"""

import numpy as np

class checkers_env:
    def __init__(self):
        self.size = 8
        self.reset()

    def reset(self):
        # 0: empty, 1: p1 pawn, 2: p1 queen, -1: p2 pawn, -2: p2 queen
        self.board = np.zeros((self.size, self.size), dtype=int)
        for i in range(3):
            for j in range(self.size):
                if (i + j) % 2 == 1:
                    self.board[i, j] = -1
        for i in range(5, 8):
            for j in range(self.size):
                if (i + j) % 2 == 1:
                    self.board[i, j] = 1
        self.current_player = 1
        self.done = False
        return self.board.copy()

    def step(self, move):
        # Reward values
        REW_CAPTURE = 2
        REW_PROMOTION = 5
        REW_ILLEGAL = -3
        REW_MISS_CAPTURE = -1
        WIN_BONUS = 10
        
        reward = 0
        info = {}
        opponent_count_before = np.count_nonzero(self.board == -self.current_player)
        if self.done:
            return self.board.copy(), 0, True, info
        if isinstance(move[0], int):
            moves = [move]
        else:
            moves = move
        must_capture, capture_moves = self.any_capture_possible(self.current_player)
        is_capture = False
        # validate the sequence step by step
        temp_board = self.board.copy()
        fr, fc, _, _ = moves[0]
        temp_piece = temp_board[fr, fc]
        for idx, m in enumerate(moves):
            fr, fc, tr, tc = m
            # always check if the piece at (fr, fc) can capture on temp_board
            must_cap = False
            for dr in [-2, 2]:
                for dc in [-2, 2]:
                    ntr, ntc = fr + dr, fc + dc
                    if 0 <= ntr < self.size and 0 <= ntc < self.size:
                        if self.is_valid_move(fr, fc, ntr, ntc, True, board=temp_board):
                            must_cap = True
            for d in [-1, 1]:
                for e in [-1, 1]:
                    for dist in range(2, self.size):
                        ntr, ntc = fr + d * dist, fc + e * dist
                        if 0 <= ntr < self.size and 0 <= ntc < self.size:
                            if self.is_valid_move(fr, fc, ntr, ntc, True, board=temp_board):
                                must_cap = True
            if not self.is_valid_move(fr, fc, tr, tc, must_cap, temp_board):
                if must_cap:                # piece ignored a mandatory capture
                    reward = REW_MISS_CAPTURE
                else:                       # truly illegal (e.g. wrong direction, occupied square…)
                    reward = REW_ILLEGAL
                self.board[fr, fc] = 0
                self.current_player *= -1
                return self.board.copy(), reward, False, info
            if abs(tr - fr) > 1:
                is_capture = True
                # remove captured piece for temp_board BEFORE moving the piece
                if abs(temp_piece) == 1:
                    cap_r = fr + (tr - fr) // 2
                    cap_c = fc + (tc - fc) // 2
                    temp_board[cap_r, cap_c] = 0
                elif abs(temp_piece) == 2:
                    dr = (tr - fr) // abs(tr - fr)
                    dc = (tc - fc) // abs(tc - fc)
                    r, c = fr + dr, fc + dc
                    while r != tr and c != tc:
                        if temp_board[r, c] != 0 and np.sign(temp_board[r, c]) == -np.sign(temp_piece):
                            temp_board[r, c] = 0
                            break
                        elif temp_board[r, c] != 0:
                            break
                        r += dr
                        c += dc
            # apply move to temp_board for next validation
            temp_board[tr, tc] = temp_board[fr, fc]
            temp_board[fr, fc] = 0
            # promotion in temp_board
            if temp_piece == 1 and tr == 0:
                temp_board[tr, tc] = 2
                temp_piece = 2
            elif temp_piece == -1 and tr == 7:
                temp_board[tr, tc] = -2
                temp_piece = -2
            else:
                temp_piece = temp_board[tr, tc]
        # mandatory capture: if not a capture, remove the piece at the start of the first move
        if must_capture and not is_capture:
            fr, fc, _, _ = moves[0]
            self.board[fr, fc] = 0
            reward = REW_MISS_CAPTURE
            self.current_player *= -1
            return self.board.copy(), reward, self.done, info
        # apply the sequence to the board
        for idx, m in enumerate(moves):
            fr, fc, tr, tc = m
            piece = self.board[fr, fc]
            self.make_move(fr, fc, tr, tc)
            # handle capture for each jump
            if abs(tr - fr) > 1:
                if abs(piece) == 1:
                    cap_r = fr + (tr - fr) // 2
                    cap_c = fc + (tc - fc) // 2
                    if self.board[cap_r, cap_c] != 0 and np.sign(self.board[cap_r, cap_c]) == -np.sign(piece):
                        self.board[cap_r, cap_c] = 0
                        reward += REW_CAPTURE
                elif abs(piece) == 2:
                    dr = (tr - fr) // abs(tr - fr)
                    dc = (tc - fc) // abs(tc - fc)
                    r, c = fr + dr, fc + dc
                    captured = 0
                    while r != tr and c != tc:
                        if self.board[r, c] != 0 and np.sign(self.board[r, c]) == -np.sign(piece):
                            self.board[r, c] = 0
                            reward += REW_CAPTURE
                            captured += 1
                        elif self.board[r, c] != 0:
                            break
                        r += dr
                        c += dc
            # promotion in board
            if (piece == 1 and tr == 0) or (piece == -1 and tr == 7):
                self.board[tr, tc] = 2 if piece == 1 else -2
                reward += REW_PROMOTION
                piece = self.board[tr, tc]
        # only check if the last moved piece can capture
        last_tr, last_tc = tr, tc
        piece = self.board[last_tr, last_tc]
        if abs(piece) == 1:
            for dr in [-2, 2]:
                for dc in [-2, 2]:
                    ntr, ntc = last_tr + dr, last_tc + dc
                    if 0 <= ntr < self.size and 0 <= ntc < self.size:
                        if self.is_valid_move(last_tr, last_tc, ntr, ntc, True):
                            self.board[last_tr, last_tc] = 0
                            reward = REW_MISS_CAPTURE  # overwrite
                            self.current_player *= -1
                            return self.board.copy(), reward, self.done, info
        elif abs(piece) == 2:
            for d in [-1, 1]:
                for e in [-1, 1]:
                    for dist in range(2, self.size):
                        ntr, ntc = last_tr + d * dist, last_tc + e * dist
                        if 0 <= ntr < self.size and 0 <= ntc < self.size:
                            if self.is_valid_move(last_tr, last_tc, ntr, ntc, True):
                                self.board[last_tr, last_tc] = 0
                                reward = REW_MISS_CAPTURE  # overwrite
                                self.current_player *= -1
                                return self.board.copy(), reward, self.done, info
        opponent_count_after = np.count_nonzero(self.board == -self.current_player)

        # if the opponent's last piece was just captured, add the win bonus.
        if opponent_count_before > 0 and opponent_count_after == 0:
            reward += WIN_BONUS
            self.done = True
        else:
            self.current_player *= -1

        return self.board.copy(), reward, self.done, info

    def is_valid_move(self, fr, fc, tr, tc, must_capture, board=None):
        board = self.board if board is None else board
        if not (0 <= fr < self.size and 0 <= fc < self.size and 0 <= tr < self.size and 0 <= tc < self.size):
            return False
        piece = board[fr, fc]
        if piece == 0:
            return False
        if np.sign(piece) != self.current_player:
            return False
        if board[tr, tc] != 0:
            return False
        dr = tr - fr
        dc = tc - fc
        # pawn logic: diagonal forward move, but capture in any diagonal direction
        if abs(piece) == 1:
            direction = np.sign(piece)
            if must_capture:
                if abs(dr) == 2 and abs(dc) == 2:
                    # allow capture in any diagonal direction
                    cap_r = fr + dr // 2
                    cap_c = fc + dc // 2
                    if board[cap_r, cap_c] != 0 and np.sign(board[cap_r, cap_c]) == -direction:
                        return True
                return False
            else:
                # diagonal forward move only
                if abs(dr) == 1 and abs(dc) == 1:
                    if (direction == 1 and dr == -1) or (direction == -1 and dr == 1):
                        return True
                # diagonal capture in any direction
                if abs(dr) == 2 and abs(dc) == 2:
                    cap_r = fr + dr // 2
                    cap_c = fc + dc // 2
                    if board[cap_r, cap_c] != 0 and np.sign(board[cap_r, cap_c]) == -direction:
                        return True
            return False
        # queen logic: diagonal any direction, any distance
        if abs(piece) == 2:
            if abs(dr) == abs(dc):
                if must_capture:
                    if abs(dr) > 1:
                        step_r = dr // abs(dr)
                        step_c = dc // abs(dc)
                        r, c = fr + step_r, fc + step_c
                        found = False
                        while r != tr and c != tc:
                            if board[r, c] != 0:
                                if np.sign(board[r, c]) == -np.sign(piece) and not found:
                                    found = True
                                else:
                                    return False
                            r += step_r
                            c += step_c
                        return found
                    return False
                else:
                    if abs(dr) == 1:
                        return True
                    if abs(dr) > 1:
                        step_r = dr // abs(dr)
                        step_c = dc // abs(dc)
                        r, c = fr + step_r, fc + step_c
                        len_encountered = 0
                        enemy = False
                        while r != tr and c != tc:
                            if board[r, c] != 0:
                                len_encountered += 1
                                if np.sign(board[r, c]) == -np.sign(piece):
                                    enemy = True
                                else:
                                    enemy = False
                            r += step_r
                            c += step_c
                        return len_encountered == 0 or (len_encountered == 1 and enemy)
            return False
        return False

    def make_move(self, fr, fc, tr, tc):
        self.board[tr, tc] = self.board[fr, fc]
        self.board[fr, fc] = 0

    def has_moves(self, player):
        must_capture, _ = self.any_capture_possible(player)
        for fr in range(self.size):
            for fc in range(self.size):
                if np.sign(self.board[fr, fc]) == player:
                    piece = self.board[fr, fc]
                    if must_capture:
                        for dr in [-2, 2]:
                            for dc in [-2, 2]:
                                tr, tc = fr + dr, fc + dc
                                if 0 <= tr < self.size and 0 <= tc < self.size:
                                    cur = self.current_player
                                    self.current_player = player
                                    valid = self.is_valid_move(fr, fc, tr, tc, must_capture)
                                    self.current_player = cur
                                    if valid:
                                        return True
                        if abs(piece) == 2:
                            for d in [-1, 1]:
                                for e in [-1, 1]:
                                    for dist in range(2, self.size):
                                        tr, tc = fr + d * dist, fc + e * dist
                                        if 0 <= tr < self.size and 0 <= tc < self.size:
                                            cur = self.current_player
                                            self.current_player = player
                                            valid = self.is_valid_move(fr, fc, tr, tc, must_capture)
                                            self.current_player = cur
                                            if valid:
                                                return True
                    else:
                        # diagonal forward move for pawns
                        if abs(piece) == 1:
                            for dr, dc in [(-1, -1), (-1, 1)] if player == 1 else [(1, -1), (1, 1)]:
                                tr, tc = fr + dr, fc + dc
                                if 0 <= tr < self.size and 0 <= tc < self.size:
                                    cur = self.current_player
                                    self.current_player = player
                                    valid = self.is_valid_move(fr, fc, tr, tc, must_capture)
                                    self.current_player = cur
                                    if valid:
                                        return True
                        if abs(piece) == 2:
                            for d in [-1, 1]:
                                for e in [-1, 1]:
                                    for dist in range(1, self.size):
                                        tr, tc = fr + d * dist, fc + e * dist
                                        if 0 <= tr < self.size and 0 <= tc < self.size:
                                            cur = self.current_player
                                            self.current_player = player
                                            valid = self.is_valid_move(fr, fc, tr, tc, must_capture)
                                            self.current_player = cur
                                            if valid:
                                                return True
        return False

    def any_capture_possible(self, player):
        capture_moves = []
        cur_player = self.current_player
        self.current_player = player
        for fr in range(self.size):
            for fc in range(self.size):
                if np.sign(self.board[fr, fc]) == player:
                    piece = self.board[fr, fc]
                    if abs(piece) == 1:
                        for dr in [-2, 2]:
                            for dc in [-2, 2]:
                                tr, tc = fr + dr, fc + dc
                                if 0 <= tr < self.size and 0 <= tc < self.size:
                                    if self.is_valid_move(fr, fc, tr, tc, True):
                                        capture_moves.append((fr, fc, tr, tc))
                    if abs(piece) == 2:
                        for d in [-1, 1]:
                            for e in [-1, 1]:
                                for dist in range(2, self.size):
                                    tr, tc = fr + d * dist, fc + e * dist
                                    if 0 <= tr < self.size and 0 <= tc < self.size:
                                        if self.is_valid_move(fr, fc, tr, tc, True):
                                            capture_moves.append((fr, fc, tr, tc))
        self.current_player = cur_player
        return (len(capture_moves) > 0), capture_moves 