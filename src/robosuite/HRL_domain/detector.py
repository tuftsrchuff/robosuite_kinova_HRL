import numpy as np
from robosuite.models.base import MujocoModel
from robosuite.models.grippers import GripperModel
import matplotlib.pyplot as plt


class Robosuite_Hanoi_Detector:
    def __init__(self, env):
        self.env = env
        self.objects = ['cube1', 'cube2', 'cube3']
        self.object_id = {'cube1': 'cube1_main', 'cube2': 'cube2_main', 'cube3': 'cube3_main', 'peg1': 'peg1_main', 'peg2': 'peg2_main', 'peg3': 'peg3_main'}
        self.object_areas = ['peg1', 'peg2', 'peg3']
        self.area_pos = {'peg1': self.env.pegs_xy_center[0], 'peg2': self.env.pegs_xy_center[1], 'peg3': self.env.pegs_xy_center[2]}
        self.grippers_areas = ['pick', 'drop', 'activate', 'lightswitch']
        self.grippers = ['gripper']
        self.area_size = self.env.peg_radius
        self.max_distance = 10 #max distance for the robotic arm in meters

    def at(self, obj, area, return_distance=False):
        obj_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj]]
        dist = np.linalg.norm(obj_pos - self.area_pos[area])
        if return_distance:
            return dist
        else:
            return bool(dist < self.area_size)

    def select_object(self, obj_name):
        """
        Selects the object tuple (object, object name) from self.objects using one of the strings ["milk", "bread", "cereal", "can"],
        ignoring the caps from the first letter in the self.obj_names.
        """
        obj_name = obj_name.lower()
        for obj, name in zip(self.env.objects, self.env.obj_names):
            if name.startswith(obj_name):
                return obj
        return None

    def grasped(self, obj):
        active_obj = self.select_object(obj)

        gripper = self.env.robots[0].gripper
        object_geoms = active_obj.contact_geoms

        if isinstance(object_geoms, MujocoModel):
            o_geoms = object_geoms.contact_geoms
        else:
            o_geoms = [object_geoms] if type(object_geoms) is str else object_geoms
        if isinstance(gripper, GripperModel):
            g_geoms = [gripper.important_geoms["left_fingerpad"], gripper.important_geoms["right_fingerpad"]]
        elif type(gripper) is str:
            g_geoms = [[gripper]]
        else:
            # Parse each element in the gripper_geoms list accordingly
            g_geoms = [[g_group] if type(g_group) is str else g_group for g_group in gripper]

        # Search for collisions between each gripper geom group and the object geoms group
        for g_group in g_geoms:
            if not self.env.check_contact(g_group, o_geoms):
                return False
        return True

    def over(self, gripper, obj, return_distance=False):
        obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
            if obj in self.object_areas:
                obj_pos = self.area_pos[obj]
            else:
                obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
            dist_xy = np.linalg.norm(gripper_pos[:-1] - obj_pos[:-1])
            if return_distance:
                return dist_xy
            else:
                return bool(dist_xy < 0.005)
    
    def at_grab_level(self, gripper, obj, return_distance=False):
        obj_body = self.env.sim.model.body_name2id(self.object_id[obj])
        if gripper == 'gripper':
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.env.gripper_body])
            obj_pos = np.asarray(self.env.sim.data.body_xpos[obj_body])
            dist_z = np.linalg.norm(gripper_pos[2] - obj_pos[2])
            if return_distance:
                return dist_z
            else:
                return bool(dist_z < 0.005)
    
    def on(self, obj1, obj2):
        obj1_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj1]]
        if obj2 in self.object_areas:
            obj2_pos = self.area_pos[obj2]
            #dist_xyz = np.linalg.norm(obj1_pos - obj2_pos)
            #return dist_xyz < 0.02 and obj1_pos[2] < obj2_pos[2]
        else:
            obj2_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[obj2]]
        dist_xyz = np.linalg.norm(obj1_pos - obj2_pos)
        dist_xy = np.linalg.norm(obj1_pos[:-1] - obj2_pos[:-1])
        dist_z = np.linalg.norm(obj1_pos[2] - obj2_pos[2])
        #return dist_xyz < 0.05 and obj1_pos[2] > obj2_pos[2]#dist_xyz < 0.05 and dist_xy < 0.05 and obj1_pos[2] > obj2_pos[2]
        return bool(dist_xy < 0.025 and obj1_pos[2] > obj2_pos[2] and dist_z < 0.05)
    
    def clear(self, obj):
        for other_obj in self.objects:
            if other_obj != obj:
                if self.on(other_obj, obj):
                    return False
        return True
    
    def open(self, gripper):
        if gripper == 'gripper':
            """
            Returns True if the gripper is open, False otherwise.
            """
            gripper = self.env.robots[0].gripper
            # Print gripper aperture
            left_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_left_inner_finger")])
            right_finger_pos = np.asarray(self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_right_inner_finger")])
            aperture = np.linalg.norm(left_finger_pos - right_finger_pos)
            #print(f'Gripper aperture: {aperture}')
            return bool(aperture > 0.13)
        else:
            return None
    
    def picked_up(self, obj, return_distance=False):
        active_obj = self.select_object(obj)
        z_target = self.env.table_offset[2] + 0.45
        object_z_loc = self.env.sim.data.body_xpos[self.env.obj_body_id[active_obj.name]][2]
        z_dist = z_target - object_z_loc
        if return_distance:
            return z_dist
        else:
            return bool(z_dist < 0.15)

    def get_groundings(self, as_dict=False, binary_to_float=False, return_distance=False):
            
            groundings = {}
    
            # Check if each object is in each area
            # for obj in self.objects:
            #     for area in self.object_areas:
            #         at_value = self.at(obj, area, return_distance=return_distance)
            #         if return_distance:
            #             at_value = at_value / self.max_distance  # Normalize distance
            #         if binary_to_float:
            #             at_value = float(at_value)
            #         groundings[f'at({obj},{area})'] = at_value

            # Check if the gripper is grasping each object
            for obj in self.objects:
                grasped_value = self.grasped(obj)
                if binary_to_float:
                    grasped_value = float(grasped_value)
                groundings[f'grasped({obj})'] = grasped_value

            # Check if the gripper is over each object
            for gripper in ['gripper']:
                for obj in self.objects+self.object_areas:
                    over_value = self.over(gripper, obj, return_distance=return_distance)
                    if return_distance:
                        over_value = over_value / self.max_distance
                    if binary_to_float:
                        over_value = float(over_value)
                    groundings[f'over({gripper},{obj})'] = over_value

            # Check if the gripper is at the same height as each object
            for gripper in ['gripper']:
                for obj in self.objects:
                    at_grab_level_value = self.at_grab_level(gripper, obj, return_distance=return_distance)
                    if return_distance:
                        at_grab_level_value = at_grab_level_value / self.max_distance
                    if binary_to_float:
                        at_grab_level_value = float(at_grab_level_value)
                    groundings[f'at_grab_level({gripper},{obj})'] = at_grab_level_value

            # Check if each object is on another object
            for obj1 in self.objects:
                for obj2 in self.objects+self.object_areas:
                    if obj1 != obj2:
                        on_value = self.on(obj1, obj2)
                        if binary_to_float:
                            on_value = float(on_value)
                        groundings[f'on({obj1},{obj2})'] = on_value

            # Check if each object is clear
            for obj in self.objects+self.object_areas:
                clear_value = self.clear(obj)
                if binary_to_float:
                    clear_value = float(clear_value)
                groundings[f'clear({obj})'] = clear_value
            
            # Check if the gripper is open
            gripper_open_value = self.open('gripper')
            if binary_to_float:
                gripper_open_value = float(gripper_open_value)
            groundings['open_gripper(gripper)'] = gripper_open_value

            # Check if an object has been picked up
            for obj in self.objects:
                picked_up_value = self.picked_up(obj, return_distance=return_distance)
                if return_distance:
                    picked_up_value = picked_up_value / self.max_distance
                if binary_to_float:
                    picked_up_value = float(picked_up_value)
                groundings[f'picked_up({obj})'] = picked_up_value

            #Key is grounded predicate and value is evaluation of predicate (True/False/numerical)
            return dict(sorted(groundings.items())) if as_dict else np.asarray([v for k, v in sorted(groundings.items())])
    
    def dict_to_array(self, groundings):
        return np.asarray([v for k, v in sorted(groundings.items())])
    
    def display_state(self, state=None):
        if state is None:
            state = self.get_groundings(as_dict=True, binary_to_float=True, return_distance=False)
        # Initialize the pegs and cubes
        pegs = [[], [], []]
        cubes_on_cubes = {}

        # Parse the state dictionary to determine the location of each cube
        for key, value in state.items():
            if key.startswith('on(') and value == 1.0:
                cube, obj = key[3:-1].split(',')
                if 'peg' in obj:
                    pegs[int(obj[-1])-1].append(cube)
                elif 'cube' in obj:
                    if obj not in cubes_on_cubes:
                        cubes_on_cubes[obj] = []
                    cubes_on_cubes[obj].append(cube)

        # Sort the cubes on each peg in descending order (largest on bottom)
        for peg in pegs:
            peg.sort(key=lambda x: int(x[-1]), reverse=True)

        # Create the plot
        fig, ax = plt.subplots()

        # Draw the pegs
        for i in range(3):
            ax.plot([i, i], [0, 3], color='black')

        # Helper function to draw a cube and any cubes on top of it
        def draw_cube(i, y, cube):
            size = int(cube[-1]) / 5.0  # Adjust the size based on the cube number
            ax.add_patch(plt.Rectangle((i-size/2, y), size, 0.5, color='blue'))
            # Draw the cubes that are on this cube
            if cube in cubes_on_cubes:
                for on_cube in cubes_on_cubes[cube]:
                    y = draw_cube(i, y+0.5, on_cube)
            return y

        # Draw the cubes
        for i, peg in enumerate(pegs):
            y = 0
            for cube in peg:
                y = draw_cube(i, y, cube) + 0.5

        # Set the x and y limits
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 2)

        # Remove the axes for a cleaner look
        ax.axis('off')

        # Show the plot
        plt.show()