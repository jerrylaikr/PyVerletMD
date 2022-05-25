import numpy as np


class Atom:
    """
    Class for particles in the simulation.
    """

    def __init__(self, pos: list[float], vel: list[float], mass: float):
        """
        Init method.

        Parameters:
            pos (list): shape=(2,)
                Initial position vector of this atom. (pos_x, pos_y)
            vel (list): shape=(2,)
                Initial velocity vector of this atom. (vel_x, vel_y)
            mass (float):
                Mass of atom in unit [eV*A^2*ps^-2]
                1[kg] = 6.242e22[eV*A^2*ps^-2]
        """
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

        # trajectory
        self.traj = np.array([pos])

        # flag to check if properties have been updated
        self._updated = {
            "force": False,
            "vel": True,
            "acc": False,
            "PE": False,
        }
        self._updated_counter = {
            "force": 0,
            "PE": 0,
        }

    def get_KE(self):
        """
        Return kinetic energy of this atom in unit [eV].
        KE = (1/2) mv^2
        """
        if not self._updated["vel"]:
            raise RuntimeError(
                "velocity has not been updated, unable to get kinetic energy"
            )
        return 0.5 * self.mass * (np.linalg.norm(self.vel) ** 2)

    def reset_force(self):
        """Reset force vector on this atom to 0."""
        self.force = np.zeros(2)
        self._updated_counter["force"] = 0

    def add_force(self, force_by_atom_j: np.ndarray):
        """
        Add force vector exerted by atom_j on this atom.

        Parameters:
            force_by_atom_j (ndarray): shape=(2,)
                Force vector exerted by atom_j on this atom.
        """
        self.force += force_by_atom_j
        self._updated_counter["force"] += 1

    def reset_PE(self):
        """Reset potential energy of this atom to 0."""
        self.PE = 0
        self._updated_counter["PE"] = 0

    def add_PE(self, PE_with_atom_j: float):
        """
        Add potential energy raised from interaction between atom_j and this atom.

        Parameters:
            PE_with_atom_j (float): Potential energy raised from interaction between atom_j and this atom.
        """
        self.PE += PE_with_atom_j
        self._updated_counter["PE"] += 1

    def update_acc(self):
        """
        Update acceleration vector of this atom at current timestep.
        Must be executed after updating force.
        a = F/m
        """
        if not self._updated["force"]:
            raise RuntimeError(
                "Force has not been updated, unable to update acceleration"
            )
        self.acc = self.force / self.mass
        self._updated["acc"] = True

    def update_pos(self, dt: float, size: np.ndarray):
        """
        Update position vector of this atom to the next timestep.
        Should only be executed after updating force, acceleration, and velocity.
        Verlet algorithm:
            r(t+dt) = 2r(t) - r(t-dt) + a * dt^2

        Parameters:
            dt (float): Size of timestep in unit [ps].
            size (np.ndarray): shape=(2,)
                Size of the simulation box in unit [A].
        """
        # check if force, acceleration, and velocity have been updated
        for prop in ("force", "acc", "vel"):
            if not self._updated[prop]:
                raise RuntimeError(
                    f"{prop} has not been updated, unable to update position"
                )
        pos_temp = np.array(self.pos)  # later this will be the prev position

        # use Verlet algorithm
        self.pos = 2 * self.pos - self.pos_prev + (dt**2 * self.acc)

        self.pos = self.pos % size  # correction for periodic boundary condition
        self.pos_prev = np.array(pos_temp)
        # append new position to trajectory
        self.traj = np.append(self.traj, [self.pos], axis=0)

        # reset "updated" flags
        self._updated["force"] = False
        self._updated["acc"] = False
        self._updated["vel"] = False

    def update_vel(self, dt: float, size: np.ndarray):
        """
        Update velocity vector of this atom at current timestep.
        Should only be executed after updating acceleration.
        Should not be executed at t=0.
        v(t) = (r(t) - r(t-dt)) / dt + dt * a(t) / 2

        Parameters:
            dt (float): Size of timestep in unit [ps].
            size (np.ndarray): shape=(2,)
                Size of the simulation box in unit [A].
        """
        if not self._updated["acc"]:
            raise RuntimeError(
                "Acceleration has not been updated, unable to update velocity"
            )

        # calculate displacement from previous position
        # PBC should be considered
        disp = (self.pos - self.pos_prev) - size * np.round(
            (self.pos - self.pos_prev) / size
        )

        # calculate velocity vector at current timestep
        self.vel = disp / dt + dt * self.acc / 2
        self._updated["vel"] = True


class Potential:
    """
    Base class for interatomic potential profiles.
    """

    def __init__(self, R_1, R_C):
        """
        Init method.

        Parameters:
            R_1 (float): transition radius in unit [A].
            R_C (float): cut-off radius in unit [A].
        """
        self.R_1 = R_1
        self.R_C = R_C

    def calc_force_btwn(self, atom_i: Atom, atom_j: Atom, box_size) -> np.ndarray:
        """
        Class method to calculate force exerted BY atom_j ON atom_i.

        Parameters:
            atom_i (Atom): The atom experiencing force.
            atom_j (Atom): The atom exerting force.
            box_size (np.ndarray): shape=(2,)
                Size of the simulation box in unit [A].

        Return:
            (np.ndarray): shape=(2,)
                Force vector exerted BY atom_j ON atom_i.
        """
        return np.zeros(2)

    def calc_PE_btwn(self, atom_i: Atom, atom_j: Atom, box_size) -> float:
        """
        Class method to calculate potential energy raised from interaction between atom_i and atom_j.

        Parameters:
            atom_i (Atom): one atom in the system.
            atom_j (Atom): another atom in the system.
            box_size (np.ndarray): shape=(2,)
                Size of the simulation box in unit [A].

        Return:
            (float): Potential energy raised from interaction between atom_i and atom_j.
        """
        return 0


class Dummy_LJ_potential(Potential):
    """
    Dummy potential class using Lennard-Jones potential.
    """

    def __init__(self, R_1=7.0, R_C=7.5):
        super().__init__(R_1=R_1, R_C=R_C)

    def calc_force_btwn(self, atom_i: Atom, atom_j: Atom, box_size) -> np.ndarray:
        """
        Distance less than transition radius:
            F = -dV/dr = -(-38608.67*r^-13 + 68.744*r^-7)
            F_x = -dV/dr * (r_x / r)
        Distance greater than transition radius but less than cut-off radius (use tail function):
            F = -dV_tail/dr = -(-0.0036 * (r-R_C)^2 - 0.0001994 * (r-R_C))
            F_x = -dV_tail/dr * (r_x / r)
        """
        # TODO: make it a real class for potential to autoatically calculate coefficients

        # calculate distance vector cosidering PBC
        r_ij_vec = (atom_i.pos - atom_j.pos) - box_size * np.round(
            (atom_i.pos - atom_j.pos) / box_size
        )

        # calculate magnitude of distance
        r_ij_mag = np.linalg.norm(r_ij_vec)

        if (r_ij_mag > 0) and (r_ij_mag <= self.R_1):
            return -(-38608.67 * r_ij_mag ** (-13) + 68.744 * r_ij_mag ** (-7)) * (
                r_ij_vec / r_ij_mag
            )
        elif (r_ij_mag > self.R_1) and (r_ij_mag < self.R_C):
            return -(
                -0.0036 * (r_ij_mag - self.R_C) ** 2 - 0.0001994 * (r_ij_mag - self.R_C)
            ) * (r_ij_vec / r_ij_mag)
        else:
            return np.zeros(2)

    def calc_PE_btwn(self, atom_i: Atom, atom_j: Atom, box_size) -> float:
        """
        Distance less than transition radius:
            Ep_LJ = 4*0.0102*((2.559/r)^12-(2.559/r)^6)
        Distance greater than transition radius but less than cut-off radius (use tail function):
            Ep_cut = (-0.00122)*(r-R_C)^3+(-9.9968e-04)*(r-R_C)^2
        """
        # calculate distance vector cosidering PBC
        r_ij_vec = (atom_i.pos - atom_j.pos) - box_size * np.round(
            (atom_i.pos - atom_j.pos) / box_size
        )

        # magnitude of distance
        r_ij_mag = np.linalg.norm(r_ij_vec)

        if 0 < r_ij_mag <= self.R_1:
            # use original potential function
            return 4 * 0.0102 * ((2.559 / r_ij_mag) ** 12 - (2.559 / r_ij_mag) ** 6)
        elif self.R_1 < r_ij_mag < self.R_C:
            # use tail function
            return (-0.00122) * (r_ij_mag - self.R_C) ** 3 + (-9.9968e-04) * (
                r_ij_mag - self.R_C
            ) ** 2
        else:
            # cutoff
            return 0


class Many_body_system:
    """
    Class for the simulation system.
    """

    def __init__(self, size: list, potential_profile: Potential, dt: float):
        """
        Init method.

        Parameters:
            size (list): shape=(2,)
                Size of the simulation box in unit [A].
            potential_profile (Potential):
                Potential object that describes the interactive potential between atoms in this system.
            dt (float): Size of timestep in unit [ps].
        """
        self.size: np.ndarray = np.array(size)
        self.potential_profile: Potential = potential_profile

        self.n_atoms: int = 0
        self.atoms_list: list[Atom] = []
        self.dt: float = dt
        self.time_elapsed: float = 0.0

        # flag to check if state at t=-dt has been evaluated
        self.neg_step_evaluated = False

    def add_atom(self, atom_pos: list, atom_vel: list, atom_mass: float):
        """
        Add new atom to the system.

        Parameters:
            atom_pos (list): shape=(2,)
                Initial position vector of this atom. (pos_x, pos_y)
            atom_vel (list): shape=(2,)
                Initial velocity vector of this atom. (vel_x, vel_y)
            atom_mass (float):
                Mass of atom in unit [eV*A^2*ps^-2]
                1[kg] = 6.242e22[eV*A^2*ps^-2]
        """
        if not (0 <= atom_pos[0] <= self.size[0] and 0 <= atom_pos[1] <= self.size[1]):
            raise ValueError("atom out of simulation box")

        self.atoms_list.append(Atom(atom_pos, atom_vel, atom_mass))
        self.n_atoms += 1

    def update_all_except_pos(self):
        """Update force, velocity, acceleration, and potential energy of all atoms."""
        # reset force and PE to zero
        for atom in self.atoms_list:
            atom.reset_force()
            atom.reset_PE()

        # add force pairs and PEs
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                atom_i, atom_j = self.atoms_list[i], self.atoms_list[j]
                force = self.potential_profile.calc_force_btwn(
                    atom_i, atom_j, self.size
                )
                atom_i.add_force(force)
                atom_j.add_force(-force)
                PE = self.potential_profile.calc_PE_btwn(atom_i, atom_j, self.size)
                atom_i.add_PE(PE)
                atom_j.add_PE(PE)

        # update accelerations and velocities
        for atom in self.atoms_list:
            if atom._updated_counter["force"] == self.n_atoms - 1:
                atom._updated["force"] = True
            if atom._updated_counter["PE"] == self.n_atoms - 1:
                atom._updated["PE"] = True
            atom.update_acc()
            if not atom._updated["vel"]:
                # do not update velocity at t=0
                atom.update_vel(self.dt, self.size)

    def next_step(self):
        """
        Update the system to next timestep.
        First update positions of all atoms in the system.
        Then update other properties according to the updated positions.
        We want to be able to extract the properties AT each timestep,
        instead of those at previous timestep
        At t=0, properties are updated by calling eval_prev_pos() before moving to next timestep.
        """
        # check if state at t=-dt has been evaluated
        if not self.neg_step_evaluated:
            self.eval_prev_pos()

        for atom in self.atoms_list:
            atom.update_pos(self.dt, self.size)

        self.update_all_except_pos()
        self.time_elapsed += self.dt

    def eval_prev_pos(self):
        """
        Evaluate position of all atoms 1 timestep before the initial state.
        r(-dt) = r(0) - v(0) * dt - (dt^2 / 2) * a(t)
        """
        if self.neg_step_evaluated:
            # this method should only be called once
            # raise RuntimeError("This method shouldn't be called if state at t=-dt has been evaluated")
            return

        self.update_all_except_pos()

        for atom in self.atoms_list:
            atom.pos_prev = (
                atom.pos - atom.vel * self.dt - (self.dt**2 / 2) * atom.acc
            )

        # set flag
        self.neg_step_evaluated = True

    def get_system_KE(self):
        """Return total kinetic energy of all atoms in unit [eV]."""
        return sum([atom.get_KE() for atom in self.atoms_list])

    def get_system_PE(self):
        """Return total potential energy of all atoms in unit [eV]."""
        return sum([atom.PE for atom in self.atoms_list])


def main():
    """
    Test script.
    Typical steps of usage:
    - initialize simulation box
    - add atoms
    - evaluate prev positions
    - for timestep in timesteps:
        - output info at current timestep:
            position of all atoms
            PE and KE of the system
            trajectory of all atoms
            ...
        - update system to next timestep
    """

    # ===Params===
    R_1 = 7
    R_C = 7.5
    MASS = (
        64 / 1000 / (6.02214076e23) * 6.242e22
    )  # 1[kg]/atom = 6.242e22[eV*A^2*ps^-2]/atom
    dt = 0.2
    n_steps = 30
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

    # check if the velocity vectors are incorrectly updated at t=0
    for atom in sim.atoms_list:
        print(f"{atom.vel = }")

    # check for energy conservation
    inital_total_energy = sim.get_system_KE() + sim.get_system_PE()
    print(f"initial total energy = {inital_total_energy}")

    # start loop
    for idx in range(n_steps + 1):
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
