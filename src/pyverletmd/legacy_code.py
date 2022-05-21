import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


# ===Define Constants===
EPSILON = 0.0102
R_1 = 7
R_C = 7.5
MASS = (
    64 / 1000 / (6.02214076e23) * 6.242e22
)  # 1[kg]/atom = 6.242e22[eV*A^2*ps^-2]/atom


# ===Define Simulation Parameters===
L_X = L_Y = 30
BOX = np.array([L_X, L_Y])
dt = 0.002
# dt = 0.02 # FOR TESTING
n_steps = 3000
# n_steps = 300 # FOR TESTING
max_array_size = n_steps + 2


# ===Define Initial Condition===
pos_ini = np.array([2, 4, 15, 10, 9, 6, 18, 11, 20, 20, 24, 0])
vel_ini = np.array(
    [-1.98, -1.24, -2.38, -2.02, 3.08, 2.52, 2.64, -3.0, -2.6, 1.74, 1.24, 2.0]
)
# n_atoms = round(len(pos_ini)/2)
n_atoms = 6  # FOR TESTING


# ===Initialize Plot Canvas===
plt.style.use("dark_background")
fig = plt.figure()
ax = plt.axes(xlim=(0, L_X), ylim=(0, L_Y), aspect="equal")
time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
debug_text = ax.text(0.02, 0.75, "", transform=ax.transAxes, fontsize=8)

traj = []
mark = []
vel_vector = []
force_vector = []
for i in range(n_atoms):
    traj.extend(ax.plot([], [], "c."))
    mark.extend(ax.plot([], [], "co"))
    vel_vector.extend(ax.plot([], [], "r-"))
    force_vector.extend(ax.plot([], [], "y-"))


# ===Initialization Function===
def init():
    for i in range(n_atoms):
        traj[i].set_data([], [])
        mark[i].set_data([], [])
        vel_vector[i].set_data([], [])
        force_vector[i].set_data([], [])

    time_text.set_text("")
    debug_text.set_text("")

    return traj + mark + vel_vector + force_vector + [time_text, debug_text]


# ===Animation Function===
def animate(frame_idx):
    idx = frame_idx + 1
    # print("\n\n\nPlotting: frame={}, idx={}, t={:.2f}ps".format(frame_idx,idx,frame_idx*dt)) # FOR TESTING
    time_text.set_text(
        "frame = {}, time = {:.2f}ps / {:.2f}ps".format(
            frame_idx, frame_idx * dt, n_steps * dt
        )
    )

    # Evaluate pos_big_array[idx] here
    update_pos_all(idx)
    debug_text.set_text("idx = {}\n".format(idx))

    # Save data for trajectory tail
    traj_len = n_steps  # // 20
    start = max((idx - traj_len, 1))
    x_traj = pos_big_array[idx : start : -(traj_len // 30), 0::2].copy()
    y_traj = pos_big_array[idx : start : -(traj_len // 30), 1::2].copy()

    # # Save data for velocity arrow
    x_velvec = np.zeros((2, n_atoms))
    y_velvec = np.zeros((2, n_atoms))
    for i in range(n_atoms):
        x_velvec[0][i] = atoms[i].pos[0]
        x_velvec[1][i] = atoms[i].pos[0] + atoms[i].vel[0]
        y_velvec[0][i] = atoms[i].pos[1]
        y_velvec[1][i] = atoms[i].pos[1] + atoms[i].vel[1]

    # # Save data for force arrow
    x_forc = np.zeros((2, n_atoms))
    y_forc = np.zeros((2, n_atoms))
    for i in range(n_atoms):
        x_forc[0][i] = atoms[i].pos[0]
        x_forc[1][i] = atoms[i].pos[0] + 10 * atoms[i].force[0]
        y_forc[0][i] = atoms[i].pos[1]
        y_forc[1][i] = atoms[i].pos[1] + 10 * atoms[i].force[1]

    # Save data for marker
    x_mark = pos_big_array[idx, 0::2].copy()
    y_mark = pos_big_array[idx, 1::2].copy()

    # print("idx={}\nx_traj={}\ny_traj={}".format(idx, x_traj, y_traj)) # FOR TESTING
    # print(pos_big_array[n_steps:max_array_size])

    # Create 2Dlines: traj and mark, and add annotation
    annot = []
    for i in range(n_atoms):
        traj[i].set_data(x_traj[:, i], y_traj[:, i])
        mark[i].set_data(x_mark[i], y_mark[i])
        vel_vector[i].set_data(x_velvec[:, i], y_velvec[:, i])
        force_vector[i].set_data(x_forc[:, i], y_forc[:, i])
        # print(str(x_mark[i]) + "," + str(y_mark[i]))
        # Add annotation of atom id
        annot_text = str("Atom[{}]".format(i))
        annot.append(
            ax.text(x_mark[i], y_mark[i] + 0.4, annot_text, color="c", fontsize=8)
        )

    return traj + mark + vel_vector + force_vector + annot + [time_text, debug_text]


# ===Define Atom Class===
class Atom:
    def __init__(self, pos, vel, dt, mass, n_atoms, atoms_list, pos_array):
        self.pos = pos
        self.vel = vel
        self.dt = dt
        self.mass = mass
        self.force = np.array([0.0, 0.0])
        self.acc = np.array([0.0, 0.0])
        self.dist = np.zeros((n_atoms, 2))
        self.atoms_list = atoms_list
        self.pos_prev = np.array([0.0, 0.0])

    def update_dist(self):
        for j in range(n_atoms):
            # self is atoms[i], interacting with atoms[j]
            self.dist[j] = (self.pos - self.atoms_list[j].pos) - BOX * np.round(
                (self.pos - self.atoms_list[j].pos) / BOX
            )
        # print("dist array={}".format(self.dist)) # FOR TESTING

    def evaluate_force(self):
        r_ij_vec = np.zeros(2)
        r_ij_mag = 0
        force_temp = np.array([0.0, 0.0])
        for j in range(n_atoms):
            # self is atoms[i], interacting with atoms[j]
            r_ij_vec = self.dist[j]
            r_ij_mag = np.linalg.norm(r_ij_vec)
            if (r_ij_mag > 0) and (r_ij_mag <= R_1):
                force_temp += -(
                    -38608.67 * r_ij_mag ** (-13) + 68.744 * r_ij_mag ** (-6)
                ) * (r_ij_vec / r_ij_mag)
            elif (r_ij_mag > R_1) and (r_ij_mag < R_C):
                force_temp += -(
                    -0.0036 * (r_ij_mag - R_C) ** 2 - 0.0001994 * (r_ij_mag - R_C)
                ) * (r_ij_vec / r_ij_mag)
        return force_temp

    def update_force(self):
        self.force = self.evaluate_force()

    def evaluate_acc(self):
        return self.evaluate_force() / self.mass

    def update_acc(self):
        self.acc = self.evaluate_acc()

    def update_vel(self):
        self.vel += self.acc * self.dt

    def update_pos(self):
        # Please update distance vector before calling this method
        self.update_force()
        self.update_acc()
        self.update_vel()
        # self.pos += self.vel * self.dt # r = r + v * dt
        self.pos = 2 * self.pos - self.pos_prev + (self.dt**2 * self.acc)
        self.pos = self.pos % BOX  # Periodic Boundary


# ===Initialize Big Array For Position Storage===
pos_big_array = np.zeros((max_array_size, n_atoms * 2))
pos_big_array[1] = pos_ini[0 : 2 * n_atoms].copy()


# ===Create Atoms===
atoms = []
for i in range(n_atoms):
    indv_pos = pos_big_array[1, 2 * i : 2 * i + 2].copy()
    indv_vel = vel_ini[2 * i : 2 * i + 2].copy()
    atoms.append(Atom(indv_pos, indv_vel, dt, MASS, n_atoms, atoms, pos_big_array))


#%%
# ===Evaluate r(-dt)===
for i in range(n_atoms):
    atoms[i].update_dist()
    pos_big_array[0, 2 * i : 2 * i + 2] = (
        atoms[i].pos - atoms[i].vel * dt - (dt**2 / 2) * atoms[i].acc
    )
    pos_big_array[0, 2 * i : 2 * i + 2] = (
        pos_big_array[0, 2 * i : 2 * i + 2] % BOX
    )  # Periodic Boundary
    atoms[i].pos_prev = pos_big_array[0, 2 * i : 2 * i + 2].copy()


# ===Main Function Updating Position Array at Each Frame===
def update_pos_all(idx, n_atoms=n_atoms, atoms_list=atoms, pos_array=pos_big_array):
    # print("\nupdating idx = {}".format(idx))
    for each_atom in atoms_list:
        each_atom.update_dist()

    # print("pos_array[{}] = {}".format(idx-1,pos_array[idx-1]))

    for i in range(n_atoms):
        atoms_list[i].pos_prev = pos_array[idx - 1, 2 * i : 2 * i + 2].copy()

    # print("atoms[0].pos_prev = {}\natoms[0].pos = {}\natoms[0].acc = {}".format(atoms[0].pos_prev,atoms[0].pos,atoms[0].acc))

    for i in range(n_atoms):
        atoms_list[i].update_pos()
        pos_array[idx + 1, 2 * i : 2 * i + 2] = atoms_list[i].pos.copy()


# ===Call The Animator===
anim = animation.FuncAnimation(
    fig, animate, init_func=init, frames=n_steps, interval=1, blit=True, repeat=False
)

plt.show()
