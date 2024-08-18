#! /usr/bin/env python3

import ctypes
import os

class Opt(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_double)), ("length", ctypes.c_int)]

class RelaxedIKS(ctypes.Structure):
    pass

dir_path = os.path.dirname(os.path.realpath(__file__))
lib = ctypes.cdll.LoadLibrary(dir_path + '/../target/debug/librelaxed_ik_lib.so')

lib.relaxed_ik_new.restype = ctypes.POINTER(RelaxedIKS)
lib.solve.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
lib.solve.restype = Opt
lib.solve_position.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.solve_position.restype = Opt
lib.solve_velocity.argtypes = [ctypes.POINTER(RelaxedIKS), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.solve_velocity.restype = Opt
lib.reset.argtypes = [ctypes.POINTER(RelaxedIKS)]

class RelaxedIKRust:
    def __init__(self, setting_file_path = None):
        '''
        setting_file_path (string): path to the setting file
                                    if no path is given, the default setting file will be used
                                    /configs/settings.yaml
        '''
        if setting_file_path is None:
            self.obj = lib.relaxed_ik_new(ctypes.c_char_p())
        else:
            self.obj = lib.relaxed_ik_new(ctypes.c_char_p(setting_file_path.encode('utf-8')))
    
    def __exit__(self, exc_type, exc_value, traceback):
        lib.relaxed_ik_free(self.obj)
    
    def solve_position(self, positions, orientations, tolerances):
        '''
        Assuming the robot has N end-effectors
        positions (1D array with length as 3*N): list of end-effector positions
        orientations (1D array with length as 4*N): list of end-effector orientations (in quaternion xyzw format)
        tolerances (1D array with length as 6*N): list of tolerances for each end-effector (x, y, z, rx, ry, rz)
        '''
        pos_arr = (ctypes.c_double * len(positions))()
        quat_arr = (ctypes.c_double * len(orientations))()
        tole_arr = (ctypes.c_double * len(tolerances))()
        for i in range(len(positions)):
            pos_arr[i] = positions[i]
        for i in range(len(orientations)):
            quat_arr[i] = orientations[i]
        for i in range(len(tolerances)):
            tole_arr[i] = tolerances[i]
        xopt = lib.solve_position(self.obj, pos_arr, len(pos_arr), quat_arr, len(quat_arr), tole_arr, len(tole_arr))
        return xopt.data[:xopt.length]

    def solve_position_with_waist(self, positions, orientations, tolerances, limits):
        self.update_torso_joint_limits(limits)
        ik_solution = self.solve_position(
            positions, orientations, tolerances)

        min_waist_angle = min(ik_solution[2], ik_solution[12])
        limits[2] = min_waist_angle - 0.001
        limits[12] = min_waist_angle - 0.001
        limits[22] = min_waist_angle + 0.001
        limits[32] = min_waist_angle + 0.001
        self.update_torso_joint_limits(limits)

        ik_solution = self.solve_position(
            positions, orientations, tolerances)
        return ik_solution
    
    def solve_velocity(self, linear_velocities, angular_velocities, tolerances):
        '''
        Assuming the robot has N end-effectors
        linear_velocities (1D array with length as 3*N): list of end-effector linear velocities
        angular_velocities (1D array with length as 4*N): list of end-effector angular velocities
        tolerances (1D array with length as 6*N): list of tolerances for each end-effector (x, y, z, rx, ry, rz)
        '''
        linear_arr = (ctypes.c_double * len(linear_velocities))()
        angular_arr = (ctypes.c_double * len(angular_velocities))()
        tole_arr = (ctypes.c_double * len(tolerances))()
        for i in range(len(linear_velocities)):
            linear_arr[i] = linear_velocities[i]
        for i in range(len(angular_velocities)):
            angular_arr[i] = angular_velocities[i]
        for i in range(len(tolerances)):
            tole_arr[i] = tolerances[i]
        xopt = lib.solve_velocity(self.obj, linear_arr, len(linear_arr), angular_arr, len(angular_arr), tole_arr, len(tole_arr))
        return xopt.data[:xopt.length]
    
    def reset(self, joint_state):
        js_arr = (ctypes.c_double * len(joint_state))()
        for i in range(len(joint_state)):
            js_arr[i] = joint_state[i]
        lib.reset(self.obj, js_arr, len(js_arr))

    def update_torso_joint_limits(self, joint_state):
        js_arr = (ctypes.c_double * len(joint_state))()
        for i in range(len(joint_state)):
            js_arr[i] = joint_state[i]
        lib.update_torso_joint_limits(self.obj, js_arr, len(js_arr))

if __name__ == '__main__':
    import numpy as np

    os.chdir("..")
    setting_file_path = '/home/summer/Documents/Github/autolife/relaxed_ik_core/configs/robot_v0_5.yaml'
    relaxed_ik = RelaxedIKRust(setting_file_path)

    poses = [0.2, 0.08, 0.672, -0.32, -0.36, -0.6, 0.62,
             -0.2, 0.07, 0.666, -0.41, -0.26, -0.64, 0.59]
    positions = []
    orientations = []
    tolerances = [0.08, 0.08, 0.08, 0., 0., 0.,
                  0.08, 0.08, 0.08, 0., 0., 0.]

    positions.extend(poses[0:3])
    orientations.extend(poses[3:7])

    positions.extend(poses[7:10])
    orientations.extend(poses[10:14])

    ik_solution = relaxed_ik.solve_position(positions, orientations, tolerances)
    print("ik_solution", np.rad2deg(ik_solution[:8]))
    print("ik_solution", np.rad2deg(ik_solution[8:]))