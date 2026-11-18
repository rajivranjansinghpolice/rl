import matplotlib
matplotlib.use('Agg')

from chapter_13.solution import plot_figure_13_1, plot_figure_13_2

if __name__ == "__main__":
    plot_figure_13_1(save_path="fig_13_1_short_corridor_value.png")
    plot_figure_13_2(save_path="fig_13_2_reinforce_short_corridor.png", episodes=800, seed=0)
    print("Figures generated: fig_13_1_short_corridor_value.png, fig_13_2_reinforce_short_corridor.png")
