import gym
import logging
import numpy as np
import random

from .game import QuartoGame, QuartoPiece

logger = logging.getLogger(__name__)

class QuartoEnv(gym.Env):
    EMPTY = 0
    metadata = {'render.modes':['terminal']}

    action_space = None
    observation_space = None

    def reset(self, random_start=True):
        self.game = QuartoGame()
        self.turns = 0
        self.piece = None
        self.broken = False
        return self._observation

    def step(self, action):
        reward = 0
        self.turns += 1
        info = {'turn': self.turns,
                'invalid': False,
                'win': False,
                'draw': False}
        if self.done:
            logger.warn("Actually already done")
            return self._observation, reward, self.done, info

        position, next = action
        logger.debug(f"Received: position: {position}, next: {next}")

        # Process the position
        if self.piece is not None:
            # Don't play on the first turn, just save the next piece
            valid = self.game.play(self.piece, position, next)
            if not valid:
                # Invalid move
                reward = -200
                self.broken = True
                info['invalid'] = True
            elif self.game.game_over:
                # We just won !
                reward = 100
                info['win'] = True
            elif self.game.draw:
                reward = 20
                info['draw'] = True
            else:
                # We managed to play something valid
                reward = 0

        # Process the next piece
        self.piece = next

        return self._observation, reward, self.done, info

    @property
    def _observation(self):
        """ game board + next piece
        """
        return (self.game.board, self.piece)

    @property
    def done(self):
        return self.broken or self.game.game_over or self.game.draw

    def render(self, mode, **kwargs):
        for row in self.game.board:
            s = ""
            for piece in row:
                if piece is None:
                    s += ". "
                else:
                    s += str(piece) + " "
            print(s)
        print(f"Next: {self.piece}, Free: {''.join(str(p) for p in self.game.free)}")
        print()

    @property
    def legal_actions(self):
        for position in self.game.free_spots:
            if len(self.game.free) == 1:
                # The last move in case of draw must propose None
                yield position, None
            else:
                for piece in self.game.free:
                    if piece == self.piece:
                        continue
                    yield position, piece


    @staticmethod
    def pieceNum(piece):
        res = 0
        if piece.big:
            res += 1
        if piece.hole:
            res += 2
        if piece.black:
            res += 4
        if piece.round:
            res += 8
        return res

    @staticmethod
    def pieceFromStr(s):
        for i in range(16):
            if str(QuartoPiece(i)) == s:
                return i
        return None

    def __del__(self):
        self.close()

class MoveEncoderV0(gym.ActionWrapper):
    """First version of the Move Encoding wrapping

    Action is [pos, next]
    """

    def __init__(self, env:gym.Env) -> None:
        super(MoveEncoderV0, self).__init__(env)
        # action is [pos, next]
        # both are not null, they are just ignored when irrelevant
        self.action_space = gym.spaces.MultiDiscrete([16, 16])

    def action(self, action):
        return self.decode(action)

    @property
    def legal_actions(self):
        for action in self.env.legal_actions:
            yield self.encode(action)

    def decode(self, action):
        position, piece = action
        position = (position % 4, position // 4)
        if piece is not None:
            piece = QuartoPiece(piece)
        return position, piece

    def encode(self, action):
        position, piece = action
        if piece is not None:
            piece = QuartoEnv.pieceNum(piece)
        return position[0] + position[1] * 4, piece


class QuartoEnvV0(QuartoEnv):
    """ The encoding that were used by the v0 of the env
    That's a subclass and not a wrapper.
    """

    def __init__(self):
        super(QuartoEnvV0, self).__init__()

        # next piece + board (flatten) 
        self.observation_space = gym.spaces.MultiDiscrete([17] * (1+4*4))
        
        # action is [pos, next]
        # both are not null, they are just ignored when irrelevant
        self.action_space = gym.spaces.MultiDiscrete([16, 16])

    def step(self, action):
        position, next = action
        if next is not None:
            next = QuartoPiece(next)
        position = (position % 4, position // 4)
        return super(QuartoEnvV0, self).step((position, next))

    @property
    def observation(self):
        board = []
        for row in self.game.board:
            for piece in row:
                if piece is None:
                    board.append(self.EMPTY)
                else:
                    board.append(QuartoEnv.pieceNum(piece) + 1)
        if self.piece is None:
            piece = [self.EMPTY]
        else :
            piece = [QuartoEnv.pieceNum(self.piece) + 1]
        return np.concatenate((piece, board)).astype(np.int8)

    @property
    def legal_actions(self):
        for position, piece in super(QuartoEnvV0, self).legal_actions:
            x, y = position
            yield x+y*4, QuartoEnv.pieceNum(piece)

