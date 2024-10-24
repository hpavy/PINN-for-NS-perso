import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# Pour plot

def plot_flow(x,y,t,norme_vitesse,frame) :   
    plt.clf()
    time = list(set(t))
    time.sort()
    indices = np.where(t == time[frame])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tripcolor(x[indices], y[indices], norme_vitesse[indices], shading='gouraud', cmap='coolwarm')
    plt.colorbar(label='Norme vitesse')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Carte de chaleur à t_ad={time[frame]:.2f}')
    
def anim(name_file, x, y, t, norme_vitesse):
    fig = plt.figure()
    def animate(frame):
        plot_flow(x,y,t,norme_vitesse, frame)
        print(frame)
        return fig,
    ani = FuncAnimation(fig, animate, frames=np.arange(0, len(set(t))))#, repeat=False)
    ani.save(name_file, writer='pillow', fps=7)