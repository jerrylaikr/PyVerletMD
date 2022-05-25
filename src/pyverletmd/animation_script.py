import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from pyverletmd.simulation_api import Many_body_system, Dummy_LJ_potential

__author__ = "jerrylaikr"
__copyright__ = "jerrylaikr"
__license__ = "MIT"


def main():
    """
    Script to visualize the MD simulation while calculating.
    CLI is work in progress.
    Live plotting is not working on system without GUI. Please save animation instead.
    """
    # ===Params===
    R_1 = 7.0
    R_C = 7.5
    MASS = (
        64 / 1000 / (6.02214076e23) * 6.242e22
    )  # 1[kg]/atom = 6.242e22[eV*A^2*ps^-2]/atom
    dt = 0.02
    n_steps = 300
    size = [30, 30]

    # initialize simulation box
    sim = Many_body_system(
        size=size, potential_profile=Dummy_LJ_potential(R_1, R_C), dt=dt
    )
    sim.add_atom([2, 4], [-1.98, -1.24], MASS)
    sim.add_atom([15, 10], [-2.38, -2.02], MASS)
    sim.add_atom([9, 6], [3.08, 2.52], MASS)
    sim.add_atom([18, 11], [2.64, -3.0], MASS)
    sim.add_atom([20, 20], [-2.6, 1.74], MASS)
    sim.add_atom([24, 0], [1.24, 2.0], MASS)

    # evaluate prev positions
    sim.eval_prev_pos()

    # ===Initialize Plot Canvas===
    plt.style.use("dark_background")
    fig = plt.figure()
    ax = plt.axes(xlim=(0, sim.size[0]), ylim=(0, sim.size[1]), aspect="equal")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    debug_text = ax.text(0.02, 0.75, "", transform=ax.transAxes, fontsize=8)

    n_atoms = sim.n_atoms
    traj = []  # for plotting atom trajectories
    mark = []  # for plotting atom positions
    vel_vector = []  # for plotting velocity vectors
    force_vector = []  # for plotting force vectors
    annot = []  # for marking atom indices
    for _ in range(n_atoms):
        traj.extend(ax.plot([], [], "c.", markersize=3))
        mark.extend(ax.plot([], [], "co", markersize=8))
        vel_vector.extend(ax.plot([], [], "r-"))
        force_vector.extend(ax.plot([], [], "y-"))
        annot.append(ax.text(0, 0.4, "", color="c", fontsize=8))

    # ===Initialization Function===
    def init():
        for i in range(n_atoms):
            traj[i].set_data([], [])
            mark[i].set_data([], [])
            vel_vector[i].set_data([], [])
            force_vector[i].set_data([], [])
            annot[i].set_text("")

        time_text.set_text("")
        debug_text.set_text("")

        return traj + mark + vel_vector + force_vector + annot + [time_text, debug_text]

    # ===Animation Function===
    def animate(frame_idx):
        print(
            "Plotting: frame={}, t={:.3f}ps".format(frame_idx, frame_idx * dt)
        )  # FOR TESTING
        time_text.set_text(
            "frame = {}, time = {:.3f}ps / {:.3f}ps".format(
                frame_idx, frame_idx * dt, n_steps * dt
            )
        )

        debug_text.set_text(
            "time_elapsed = {:.3f}\nKE = {:.5f}\nPE = {:.5f}".format(
                sim.time_elapsed, sim.get_system_KE(), sim.get_system_PE()
            )
        )

        # Save data for trajectory tail
        traj_len = 100
        traj_interval = 10
        start_idx = max((traj_interval * ((frame_idx - traj_len) // traj_interval), 0))
        x_traj = np.array(
            [atom.traj[start_idx::traj_interval, 0] for atom in sim.atoms_list]
        )
        y_traj = np.array(
            [atom.traj[start_idx::traj_interval, 1] for atom in sim.atoms_list]
        )

        # # Save data for velocity arrow
        x_velvec = np.zeros((2, n_atoms))
        y_velvec = np.zeros((2, n_atoms))
        for i in range(n_atoms):
            x_velvec[0][i] = sim.atoms_list[i].pos[0]
            x_velvec[1][i] = sim.atoms_list[i].pos[0] + sim.atoms_list[i].vel[0]
            y_velvec[0][i] = sim.atoms_list[i].pos[1]
            y_velvec[1][i] = sim.atoms_list[i].pos[1] + sim.atoms_list[i].vel[1]

        # # Save data for force arrow
        x_forc = np.zeros((2, n_atoms))
        y_forc = np.zeros((2, n_atoms))
        for i in range(n_atoms):
            x_forc[0][i] = sim.atoms_list[i].pos[0]
            x_forc[1][i] = sim.atoms_list[i].pos[0] + 10 * sim.atoms_list[i].force[0]
            y_forc[0][i] = sim.atoms_list[i].pos[1]
            y_forc[1][i] = sim.atoms_list[i].pos[1] + 10 * sim.atoms_list[i].force[1]

        # Save data for marker
        x_mark = np.array([atom.pos[0] for atom in sim.atoms_list])
        y_mark = np.array([atom.pos[1] for atom in sim.atoms_list])

        # print("idx={}\nx_traj={}\ny_traj={}".format(idx, x_traj, y_traj)) # FOR TESTING
        # print(pos_big_array[n_steps:max_array_size]) # FOR TESTING
        # for i in range(6): # FOR TESTING
        #     print(f"atom {i}:") # FOR TESTING
        #     print(f"\tpos: {x_mark[i]}, {y_mark[i]}") # FOR TESTING

        # Create 2Dlines: traj and mark, and add annotation
        for i in range(n_atoms):
            traj[i].set_data(x_traj[i], y_traj[i])
            mark[i].set_data(x_mark[i], y_mark[i])
            vel_vector[i].set_data(x_velvec[:, i], y_velvec[:, i])
            force_vector[i].set_data(x_forc[:, i], y_forc[:, i])
            # print(str(x_mark[i]) + "," + str(y_mark[i]))
            # Add annotation of atom id
            annot_text = str("Atom[{}]".format(i))
            annot[i].set_text(annot_text)
            annot[i].set_position((x_mark[i], y_mark[i] + 0.4))

        # update to next state
        sim.next_step()

        return traj + mark + vel_vector + force_vector + annot + [time_text, debug_text]

    # ===Call The Animator===
    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=n_steps + 1,
        interval=1,
        blit=True,
        repeat=False,
    )

    plt.show()

    # f = r""
    # writergif = animation.PillowWriter(fps=10, bitrate=50)
    # anim.save(f, writer=writergif)


if __name__ == "__main__":
    main()
