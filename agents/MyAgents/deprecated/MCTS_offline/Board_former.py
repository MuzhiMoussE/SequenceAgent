import numpy as np

class BoardGeneralize:
    def __init__(self):pass

    def rotate_board(self, board, k):
        return np.rot90(board, k)


    def flip_board_horizontal(self, board):
        return np.fliplr(board)


    def flip_board_vertical(self, board):
        return np.flipud(board)


    def flip_board_main_diagonal(self, board):
        return np.transpose(board)


    def flip_board_anti_diagonal(self, board):
        return np.fliplr(np.flipud(np.transpose(board)))


    def all_board_transforms(self, board):
        transforms = [board, self.rotate_board(board, 1), self.rotate_board(board, 2), self.rotate_board(board, 3),
                      self.flip_board_horizontal(board), self.flip_board_vertical(board),
                      self.flip_board_main_diagonal(board), self.flip_board_anti_diagonal(board)]
        return transforms

    def get_canonical_form(self, board):
        all_forms = []
        for b in self.all_board_transforms(board):
            flattened = tuple(cell for row in b for cell in row)
            all_forms.append(flattened)
        key_afterjoin = ''.join(min(all_forms))
        return key_afterjoin
