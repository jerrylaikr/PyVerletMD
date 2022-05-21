import numpy as np


class Atom:
    def __init__(self, pos: list[float], vel: list[float], mass: float):
        # position vector
        self.pos: np.ndarray = np.array(pos)

        # velocity vector
        self.vel: np.ndarray[float] = np.array(vel)

        # mass
        self.mass: float = mass

        # initialize force vector and acceleration vector
        self.force: np.ndarray = np.zeros(2)
        self.acc: np.ndarray = np.zeros(2)

        # previous position
        self.pos_prev: np.ndarray = np.zeros(2)

    def get_KE(self):
        return 0.5 * self.mass * (self.vel[0] ** 2 + self.vel[1] ** 2)

    def reset_force(self):
        self.force = np.zeros(2)

    def add_force(self, force_by_atom_j):
        self.force += force_by_atom_j

    def update_acc(self):
        self.acc = self.force / self.mass

    def update_pos(self, dt, size):
        pos_temp = np.array(self.pos)  # later this will be the prev position
        self.pos = 2 * self.pos - self.pos_prev + (dt**2 * self.acc)
        self.pos = self.pos % size  # Periodic Boundary Condition
        self.pos_prev = np.array(pos_temp)

    def update_vel(self, dt):
        self.vel += self.acc * dt


class Potential:
    def __init__(self, R_1=7.0, R_C=7.5):
        # R_1: transition distance
        # R_C: cut-off distance
        self.R_1 = R_1
        self.R_C = R_C

    def get_force(self, atom_i: Atom, atom_j: Atom, box_size) -> np.ndarray:
        # force exerted by atom_j on atom_i
        # TODO: make it a real class for potential to autoatically calculate coefficients

        # distance vector cosidering PBC
        r_ij_vec = (atom_i.pos - atom_j.pos) - box_size * np.round(
            (atom_i.pos - atom_j.pos) / box_size
        )

        # magnitude of distance
        r_ij_mag = np.linalg.norm(r_ij_vec)

        if (r_ij_mag > 0) and (r_ij_mag <= self.R_1):
            return -(-38608.67 * r_ij_mag ** (-13) + 68.744 * r_ij_mag ** (-6)) * (
                r_ij_vec / r_ij_mag
            )
        elif (r_ij_mag > self.R_1) and (r_ij_mag < self.R_C):
            return -(
                -0.0036 * (r_ij_mag - self.R_C) ** 2 - 0.0001994 * (r_ij_mag - self.R_C)
            ) * (r_ij_vec / r_ij_mag)
        else:
            return np.zeros(2)


class Many_body_system:
    def __init__(self, size: list, potential_profile: Potential, dt: float):
        self.size: np.ndarray = np.array(size)
        self.potential_profile: Potential = potential_profile

        self.n_atoms: int = 0
        self.atoms_list: list[Atom] = []
        self.dt: float = dt

    def add_atom(self, atom_pos, atom_vel, atom_mass):
        if not (0 <= atom_pos[0] <= self.size[0] and 0 <= atom_pos[1] <= self.size[1]):
            raise ValueError("atom out of simulation box")

        self.atoms_list.append(Atom(atom_pos, atom_vel, atom_mass))
        self.n_atoms += 1

    def update_forces_accs_vel(self):
        # reset force to zero
        for atom in self.atoms_list:
            atom.reset_force()

        # add force pairs
        for i in range(self.n_atoms):
            for j in range(i, self.n_atoms):
                atom_i, atom_j = self.atoms_list[i], self.atoms_list[j]
                force = self.potential_profile.get_force(atom_i, atom_j, self.size)
                atom_i.add_force(force)
                atom_j.add_force(-force)

        # update accelerations and velocities
        for atom in self.atoms_list:
            atom.update_acc()
            atom.update_vel(self.dt)

    def next_step(self):
        # first update positions
        # then update forces and accelerations
        # at t=0, forces are updated before moving to next step
        # we want to be able to extract the forces and accs AT each timestep
        # not those AT prev timestep
        for atom in self.atoms_list:
            atom.update_pos(self.dt, self.size)

        self.update_forces_accs_vel()

    def eval_prev_pos(self):
        # evaluate r(-dt)
        self.update_forces_accs_vel()

        for atom in self.atoms_list:
            atom.pos_prev = (
                atom.pos - atom.vel * self.dt - (self.dt**2 / 2) * atom.acc
            )


def main():
    """
    initialize simulation box
    add atoms

    evaluate prev positions

    for timestep in timesteps:
        output info at current timestep:
            position of all atoms
            PE and KE of the system
            trajectory of all atoms
            ...
        update system to next timestep
    """

    # ===Params===
    R_1 = 7
    R_C = 7.5
    MASS = (
        64 / 1000 / (6.02214076e23) * 6.242e22
    )  # 1[kg]/atom = 6.242e22[eV*A^2*ps^-2]/atom
    dt = 0.02
    n_steps = 500
    size = [30, 30]

    # initialize simulation box
    sim = Many_body_system(size=size, potential_profile=Potential(R_1, R_C), dt=dt)
    sim.add_atom([2, 4], [-1.98, -1.24], MASS)
    sim.add_atom([15, 10], [-2.38, -2.02], MASS)
    sim.add_atom([9, 6], [3.08, 2.52], MASS)
    sim.add_atom([18, 11], [2.64, -3.0], MASS)
    sim.add_atom([20, 20], [-2.6, 1.74], MASS)
    sim.add_atom([24, 0], [1.24, 2.0], MASS)

    # evaluate prev positions
    sim.eval_prev_pos()

    # start loop
    for idx in range(n_steps):
        # output info of current state
        print(
            "frame = {}, time = {:.3f}ps / {:.3f}ps".format(idx, idx * dt, n_steps * dt)
        )
        for i in range(sim.n_atoms):
            print(f"atom {i}:")
            atom = sim.atoms_list[i]
            print(f"\tpos: {atom.pos}")
            # print(f"\tfor: {atom.force}")

        # update to next state
        sim.next_step()


if __name__ == "__main__":
    main()
