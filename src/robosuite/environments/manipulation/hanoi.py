from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.objects import PlateVisualObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat


class Hanoi(SingleArmEnv):
    """
    This class corresponds to the Towers of Hanoi task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        table_full_size (3-tuple): x, y, and z dimensions of the table.

        table_friction (3-tuple): the three mujoco friction parameters for
            the table.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mujoco",
        renderer_config=None,
        random_reset = False,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array((0, 0, 0.8))
        self.random_reset = random_reset

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.reset_state = None
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def reward(self, action):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.0 is provided if the red block is stacked on the green block

        Un-normalized components if using reward shaping:

            - Reaching: in [0, 0.25], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube
            - Aligning: in [0, 0.5], encourages aligning one cube over the other
            - Stacking: in {0, 2}, non-zero if cube is stacked on other cube

        The reward is max over the following:

            - Reaching + Grasping
            - Lifting + Aligning
            - Stacking

        The sparse reward only consists of the stacking component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.0 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        r_reach, r_lift, r_stack = self.staged_rewards()
        if self.reward_shaping:
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 2.0 if r_stack > 0 else 0.0

        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.0

        return reward

    def staged_rewards(self):
        """
        Helper function to calculate staged rewards based on current physical states.

        Returns:
            3-tuple:

                - (float): reward for reaching and grasping
                - (float): reward for lifting and aligning
                - (float): reward for stacking
        """
        # reaching is successful when the gripper site is close to the center of the cube
        cube1_pos = self.sim.data.body_xpos[self.cube1_body_id]
        cube2_pos = self.sim.data.body_xpos[self.cube2_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.robots[0].eef_site_id]
        dist = np.linalg.norm(gripper_site_pos - cube1_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # grasping reward
        grasping_cube1 = self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.cube1)
        if grasping_cube1:
            r_reach += 0.25

        # lifting is successful when the cube is above the table top by a margin
        cube1_height = cube1_pos[2]
        table_height = self.table_offset[2]
        cube1_lifted = cube1_height > table_height + 0.04
        r_lift = 1.0 if cube1_lifted else 0.0

        # Aligning is successful when cube1 is right above cube2
        if cube1_lifted:
            horiz_dist = np.linalg.norm(np.array(cube1_pos[:2]) - np.array(cube2_pos[:2]))
            r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        # stacking is successful when the block is lifted and the gripper is not holding the object
        r_stack = 0
        cube1_touching_cube2 = self.check_contact(self.cube1, self.cube2)
        if not grasping_cube1 and r_lift > 0 and cube1_touching_cube2:
            r_stack = 2.0

        return r_reach, r_lift, r_stack

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        text_number1 = CustomMaterial(
            texture="Number1",
            tex_name="number1",
            mat_name="number1_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        text_number2 = CustomMaterial(
            texture="Number2",
            tex_name="number2",
            mat_name="number2_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        text_number3 = CustomMaterial(
            texture="Number3",
            tex_name="number3",
            mat_name="number3_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        text_number4 = CustomMaterial(
            texture="Number4",
            tex_name="number4",
            mat_name="number4_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )

        self.cube1 = BoxObject(
            name="cube1",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[1, 0, 0, 1],
            material=text_number1,
        )
        self.cube2 = BoxObject(
            name="cube2",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=text_number2,
        )
        self.cube3 = BoxObject(
            name="cube3",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=text_number3,
        )
        self.cube4 = BoxObject(
            name="cube4",
            size_min=[0.025, 0.025, 0.025],
            size_max=[0.025, 0.025, 0.025],
            rgba=[0, 1, 0, 1],
            material=text_number4,
        )

        # Add visual pegs
        self.visual_peg1 = PlateVisualObject(name="peg1")
        self.visual_peg2 = PlateVisualObject(name="peg2")
        self.visual_peg3 = PlateVisualObject(name="peg3")
        self.pegs_xy_pos = [[0.1, -0.13], [0.1, 0.07], [0.1, 0.27]]
        # Set pegs to be centered at xy pos (-0.1 x-axis, -0.05 y-axis shifted from the pegs_xy_pos)
        self.pegs_xy_center = [[0, -0.18, 0.8], [0, 0.02, 0.8], [0, 0.22, 0.8]]
        self.peg_radius = 0.0
        
        cubes = [self.cube1, self.cube2, self.cube3]
        pegs = [self.visual_peg1, self.visual_peg2, self.visual_peg3]
        self.objects = cubes + pegs
        self.obj_names = [obj.name for obj in self.objects]
        self.obj_map = {'cube1':self.cube1, 'cube2':self.cube2,'cube3':self.cube3,}

        # Create placement initializer
        if self.placement_initializer is not None:
            self.placement_initializer.reset()
            self.placement_initializer.add_objects(cubes)
        else:
            # If no custom placement initializer is provided, use a default sequential composite sampler
            self.peg_placement_initializer = SequentialCompositeSampler(name="PegSampler")
            self.peg_placement_initializer.append_sampler(
                    UniformRandomSampler(
                        name="PegSampler1",
                        mujoco_objects=self.visual_peg1,
                        x_range=[self.pegs_xy_pos[0][0], self.pegs_xy_pos[0][0]],
                        y_range=[self.pegs_xy_pos[0][1], self.pegs_xy_pos[0][1]],
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False,
                        reference_pos=self.table_offset,
                        z_offset=-0.062,
                    ))
            self.peg_placement_initializer.append_sampler(
                    UniformRandomSampler(
                        name="PegSampler2",
                        mujoco_objects=self.visual_peg2,
                        x_range=[self.pegs_xy_pos[1][0], self.pegs_xy_pos[1][0]],
                        y_range=[self.pegs_xy_pos[1][1], self.pegs_xy_pos[1][1]],
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False,
                        reference_pos=self.table_offset,
                        z_offset=-0.062,
                    ))
            self.peg_placement_initializer.append_sampler(
                    UniformRandomSampler(
                        name="PegSampler3",
                        mujoco_objects=self.visual_peg3,
                        x_range=[self.pegs_xy_pos[2][0], self.pegs_xy_pos[2][0]],
                        y_range=[self.pegs_xy_pos[2][1], self.pegs_xy_pos[2][1]],
                        rotation=0,
                        ensure_object_boundary_in_range=False,
                        ensure_valid_placement=False,
                        reference_pos=self.table_offset,
                        z_offset=-0.062,
                    ))

            if not(self.random_reset):
                place = 0
            else:
                place = np.random.randint(0, 3)
            self.placement_initializer1 = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube3,
                x_range=[self.pegs_xy_pos[place][0]-0.1, self.pegs_xy_pos[place][0]-0.1],
                y_range=[self.pegs_xy_pos[place][1]-0.05, self.pegs_xy_pos[place][1]-0.05],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            self.placement_initializer2 = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube2,
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            self.placement_initializer3 = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.cube1,
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
        self.object_placements = {"cube1":None, "cube2":None, "cube3":None}
        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=cubes+pegs,
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.cube1_body_id = self.sim.model.body_name2id(self.cube1.root_body)
        self.cube2_body_id = self.sim.model.body_name2id(self.cube2.root_body)

    def set_reset_state(self, state):
        self.reset_state = state

    def reset_positions(self, predicates):
        """
        Resets the cube placements given a predicate dictionnary
        For instance, if on(cube1,cube2) is True in the dict, will reset cube1 on cube2
        """
        self.object_placements = self.get_pos("cube3", predicates)
        self.object_placements.update(self.get_pos("cube2", predicates))
        self.object_placements.update(self.get_pos("cube1", predicates))

    def get_pos(self, cube, predicates):
        # Search for cube1 position
        onto_obj = None
        peg = None
        for key in predicates.keys():
            if "on" in key and cube in key.split(',')[0] and "cube" in key.split(',')[1]:
                onto_obj = key.split(',')[1][:-1]
            elif "on" in key and cube in key.split(',')[0] and "peg" in key.split(',')[1]:
                peg = int(key.split(',')[1][-2])-1
        if peg != None:
            print("Resetting {} on top of peg {}".format(cube, peg))
            placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.obj_map[cube],
                x_range=[self.pegs_xy_pos[peg][0]-0.1, self.pegs_xy_pos[peg][0]-0.1],
                y_range=[self.pegs_xy_pos[peg][1]-0.05, self.pegs_xy_pos[peg][1]-0.05],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            object_placement = placement_initializer.sample()
        if onto_obj != None:
            placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=self.obj_map[cube],
                rotation=0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            )
            object_placement = placement_initializer.sample(reference=self.object_placements[onto_obj][onto_obj][0], on_top=True)
        return object_placement

    def random_reset_f(self):
        object1_placement = self.placement_initializer1.sample()
        # list all objects that cube2 can be placed on (including pegs)
        list_choice2 = [object1_placement["cube3"][0], tuple(self.pegs_xy_center[0]), tuple(self.pegs_xy_center[1]), tuple(self.pegs_xy_center[2])]
        object_2_ontop = list_choice2[np.random.randint(0, 4)]
        object2_placement = self.placement_initializer2.sample(reference=object_2_ontop, on_top=True)
        # list all objects that cube1 can be placed on (including pegs)
        list_choice3 = [object1_placement["cube3"][0], object2_placement["cube2"][0], tuple(self.pegs_xy_center[0]), tuple(self.pegs_xy_center[1]), tuple(self.pegs_xy_center[2])]
        list_choice3.remove(object_2_ontop)
        object_3_ontop = list_choice3[np.random.randint(0, 4)]
        object3_placement = self.placement_initializer3.sample(reference=object_3_ontop, on_top=True)
        self.object_placements = object1_placement
        self.object_placements.update(object2_placement)
        self.object_placements.update(object3_placement)

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        self.obj_body_id = {
            "cube1": self.sim.model.body_name2id(self.cube1.root_body),
            "cube2": self.sim.model.body_name2id(self.cube2.root_body),
            "cube3": self.sim.model.body_name2id(self.cube3.root_body),
            "peg1": self.sim.model.body_name2id(self.visual_peg1.root_body),
            "peg2": self.sim.model.body_name2id(self.visual_peg2.root_body),
            "peg3": self.sim.model.body_name2id(self.visual_peg3.root_body),
        }
        self.gripper_body = self.sim.model.body_name2id('gripper0_eef')

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for pegs
            peg_placements = self.peg_placement_initializer.sample()
            for obj_pos, obj_quat, obj in peg_placements.values():
                self.sim.model.body_pos[self.sim.model.body_name2id(obj.root_body)] = obj_pos
                self.sim.model.body_quat[self.sim.model.body_name2id(obj.root_body)] = obj_quat

            # Sample from the placement initializer for cubes
            if self.random_reset:
                self.random_reset_f()
            elif self.reset_state != None:
                self.reset_positions(self.reset_state)
            else:
                object1_placement = self.placement_initializer1.sample()
                object2_placement = self.placement_initializer2.sample(reference=object1_placement["cube3"][0], on_top=True)
                object3_placement = self.placement_initializer3.sample(reference=object2_placement["cube2"][0], on_top=True)
                self.object_placements = object1_placement
                self.object_placements.update(object2_placement)
                self.object_placements.update(object3_placement)
            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in self.object_placements.values():
                self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        # low-level object information
        if self.use_object_obs:
            # Get robot prefix and define observables modality
            pf = self.robots[0].robot_model.naming_prefix
            modality = "object"

            # position and rotation of the first cube
            @sensor(modality=modality)
            def cube1_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube1_body_id])

            @sensor(modality=modality)
            def cube1_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube1_body_id]), to="xyzw")

            @sensor(modality=modality)
            def cube2_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.cube2_body_id])

            @sensor(modality=modality)
            def cube2_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.cube2_body_id]), to="xyzw")

            @sensor(modality=modality)
            def gripper_to_cube1(obs_cache):
                return (
                    obs_cache["cube1_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cube1_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def gripper_to_cube2(obs_cache):
                return (
                    obs_cache["cube2_pos"] - obs_cache[f"{pf}eef_pos"]
                    if "cube2_pos" in obs_cache and f"{pf}eef_pos" in obs_cache
                    else np.zeros(3)
                )

            @sensor(modality=modality)
            def cube1_to_cube2(obs_cache):
                return (
                    obs_cache["cube2_pos"] - obs_cache["cube1_pos"]
                    if "cube1_pos" in obs_cache and "cube2_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [cube1_pos, cube1_quat, cube2_pos, cube2_quat, gripper_to_cube1, gripper_to_cube2, cube1_to_cube2]
            names = [s.__name__ for s in sensors]

            # Create observables
            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )

        return observables

    def _check_success(self):
        """
        Check if blocks are stacked correctly.

        Returns:
            bool: True if blocks are correctly stacked
        """
        _, _, r_stack = self.staged_rewards()
        return r_stack > 0

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

        # Color the gripper visualization site according to its distance to the cube
        if vis_settings["grippers"]:
            self._visualize_gripper_to_target(gripper=self.robots[0].gripper, target=self.cube1)
