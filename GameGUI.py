import tkinter as tk

class GameGUI():

    def __init__(self, windowSize=None):
        self.windowSize = windowSize

        if windowSize == None:
            self.windowSize = (1100, 550)

        self.root = tk.Tk()
        self.root.geometry(f'{self.windowSize[0]}x{self.windowSize[1]}')

        agentID = 'DUMMY'
        self.root.title(f'QUARTO! - Playing against {agentID}')

    def startGUI(self):
        self.root.mainloop()

