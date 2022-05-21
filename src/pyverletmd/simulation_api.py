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

        # flag to check if properties have been updated
        self._updated = {
            "force": False,
            "vel": False,
            "acc": False,
            "PE": False,
        }
        self._updated_counter = {
            "force": 0,
            "PE": 0,
        }

    def get_KE(self):
        return (
            0.5 * self.mass * (self.vel[0] ** 2 + self.vel[1] ** 2)
        )  # np.linalg.norm(self.vel)** 2

    def reset_force(self):
        self.force = np.zeros(2)
        self._updated_counter["force"] = 0

    def add_force(self, force_by_atom_j):
        self.force += force_by_atom_j
        self._updated_counter["force"] += 1

    def reset_PE(self):
        self.PE = 0
        self._updated_counter["PE"] = 0

    def add_PE(self, PE_with_atom_j):
        self.PE += PE_with_atom_j
        self._updated_counter["PE"] += 1

    def update_acc(self):
        if not self._updated["force"]:
            raise RuntimeError("force has not been updated")
        self.acc = self.force / self.mass
        self._updated["acc"] = True

    def update_pos(self, dt, size):
        for prop in ("force", "acc", "vel"):
            if not self._updated[prop]:
                raise RuntimeError(f"{prop} has not been updated")
        pos_temp = np.array(self.pos)  # later this will be the prev position
        self.pos = 2 * self.pos - self.pos_prev + (dt**2 * self.acc)
        self.pos = self.pos % size  # Periodic Boundary Condition
        self.pos_prev = np.array(pos_temp)

        # reset "updated" flags
        self._updated["force"] = False
        self._updated["acc"] = False
        self._updated["vel"] = False

    def update_vel(self, dt):
        if not self._updated["acc"]:
            raise RuntimeError("acc has not been updated")
        self.vel += self.acc * dt
        self._updated["vel"] = True


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

    def get_PE(self, atom_i: Atom, atom_j: Atom, box_size) -> float:
        # distance vector cosidering PBC
        r_ij_vec = (atom_i.pos - atom_j.pos) - box_size * np.round(
            (atom_i.pos - atom_j.pos) / box_size
        )

        # magnitude of distance
        r_ij_mag = np.linalg.norm(r_ij_vec)

        if (r_ij_mag > 0) and (r_ij_mag <= self.R_1):
            return 4 * 0.0102 * ((2.559 / r_ij_mag) ** 12 - (2.559 / r_ij_mag) ** 6)
        elif (r_ij_mag > self.R_1) and (r_ij_mag < self.R_C):
            return (-0.00122) * (r_ij_mag - 7.5) ** 3 + (-9.9968e-04) * (
                r_ij_mag - 7.5
            ) ** 2
        else:
            return 0


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

    def update_all_except_pos(self):
        # reset force and PE to zero
        for atom in self.atoms_list:
            atom.reset_force()
            atom.reset_PE()

        # add force pairs and PEs
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                atom_i, atom_j = self.atoms_list[i], self.atoms_list[j]
                force = self.potential_profile.get_force(atom_i, atom_j, self.size)
                atom_i.add_force(force)
                atom_j.add_force(-force)
                PE = self.potential_profile.get_PE(atom_i, atom_j, self.size)
                atom_i.add_PE(PE)
                atom_j.add_PE(PE)

        # update accelerations and velocities
        for atom in self.atoms_list:
            if atom._updated_counter["force"] == self.n_atoms - 1:
                atom._updated["force"] = True
            if atom._updated_counter["PE"] == self.n_atoms - 1:
                atom._updated["PE"] = True
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

        self.update_all_except_pos()

    def eval_prev_pos(self):
        # evaluate r(-dt)
        self.update_all_except_pos()

        for atom in self.atoms_list:
            atom.pos_prev = (
                atom.pos - atom.vel * self.dt - (self.dt**2 / 2) * atom.acc
            )

    def get_system_KE(self):
        return sum([atom.get_KE() for atom in self.atoms_list])

    def get_system_PE(self):
        return sum([atom.PE for atom in self.atoms_list])


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

    inital_total_energy = sim.get_system_KE() + sim.get_system_PE()
    print(f"initial total energy = {inital_total_energy}")

    # start loop
    for idx in range(n_steps):
        # output info of current state
        print(
            "frame = {}, time = {:.3f}ps / {:.3f}ps".format(idx, idx * dt, n_steps * dt)
        )
        total_energy = sim.get_system_KE() + sim.get_system_PE()
        print(
            f"total energy = {total_energy}, ratio = {total_energy/inital_total_energy}"
        )
        # for i in range(sim.n_atoms):
        # print(f"atom {i}:")
        # atom = sim.atoms_list[i]
        # print(f"\tpos: {atom.pos}")
        # print(f"\tfor: {atom.force}")

        # update to next state
        sim.next_step()


if __name__ == "__main__":
    main()
