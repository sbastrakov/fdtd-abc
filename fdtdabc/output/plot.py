import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


class Plot:
    def __init__(self, period):
        self.period = period
        #self.fig, self.axs = plt.subplots(1, 2, constrained_layout=True)
        #self.axs[0].set_title('Ey(x, y, z_center)')
        #self.axs[0].set_xlabel('x (cells)')
        #self.axs[0].set_ylabel('y (cells)')
        #self.axs[1].set_title('Bz(x, y, z_center)')
        #self.axs[1].set_xlabel('x (cells)')
        #self.axs[1].set_ylabel('y (cells)')
        self.fig = plt.figure()
        self.images = []

    def add_frame(self, grid, iteration):
        if (self.period == 0) or (iteration % self.period != 0):
            return
        slice_z = grid.num_cells[2] // 2
        #ey = np.transpose(grid.ey[:, :, slice_z])
        #bz = np.transpose(grid.bz[:, :, slice_z])
        #self.fig.suptitle('Iteration ' + str(iteration), fontsize=16)
        #self.images.append(self.axs[0].imshow(ey))
        #self.axs[1].imshow(bz)
        ez = np.transpose(grid.ey[:, :, slice_z])
        self.images.append([plt.imshow(ez)])

    def animate(self):
        if self.period == 0:
            return
        ani = animation.ArtistAnimation(self.fig, self.images, interval=500, blit=True,
                                repeat_delay=1000)
        plt.show()
