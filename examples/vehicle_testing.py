#!/usr/bin/env python
"""
| File: 4_python_single_vehicle.py
| Author: Marcelo Jacinto and Joao Pinto (marcelo.jacinto@tecnico.ulisboa.pt, joao.s.pinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to use the control backends API to create a custom controller
for the vehicle from scratch and use it to perform a simulation, without using PX4 nor ROS.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp
import time

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": True})
# simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import (
    create_prim,
    find_matching_prim_paths,
    get_prim_at_path,
)
from omni.isaac.core.utils.rotations import euler_angles_to_quat

# from omni.isaac.core.objects import UsdObject

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.graphical_sensors.monocular_camera import MonocularCamera
from pegasus.simulator.logic.graphical_sensors.lidar import Lidar
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.graphs import ROS2CameraGraph


# Import the custom python control backend
import sys, os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + "/utils")
from nonlinear_controller import NonlinearController

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation
import numpy as np

# Use pathlib for parsing the desired trajectory from a CSV file
from pathlib import Path


class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()
        self.max_loop_cnt = 1000
        self.loop_cnt = 0

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics,
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(
            "/home/air/Downloads/isaac-sim-assets-1@4.2.0-rc.18+release.16044.3b2ed111/Assets/Isaac/4.2/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
            # "/home/air/Downloads/isaac-sim-assets-1@4.2.0-rc.18+release.16044.3b2ed111/Assets/Isaac/4.2/Isaac/Environments/Grid/default_environment.usd"
        )

        # Thêm object từ USD
        prim_path = "/World/Box"
        OBJECT_ASSETS = {
            "Wooden Container": {
                "usd_path": "/home/air/Downloads/Industrial_NVD@10012/Assets/ArchVis/Industrial/Containers/Wooden/WoodenCrate_A1.usd",
                "scale": [0.01, 0.01, 0.01],
                "position": [-3.0, 16.0, 0.01],
                "mass": 2.0,
                "trajectory_path": "/trajectories/testing.csv",
            },
            "Warehouse Pile": {
                "usd_path": "/home/air/Downloads/Industrial_NVD@10012/Assets/ArchVis/Industrial/Piles/WarehousePile_A1.usd",
                "scale": [0.01, 0.01, 0.01],
                "position": [-3.0, 18.0, 0.01],
                "mass": 2.0,
                "trajectory_path": "/trajectories/testing_1.csv",
            },
            "Warehouse Pile Simple": {
                "usd_path": "/home/air/Downloads/Industrial_NVD@10012/Assets/ArchVis/Industrial/Piles/WarehousePile_A1.usd",
                "scale": [0.01, 0.01, 0.01],
                "position": [-3.0, 18.0, 0.01],
                "mass": 2.0,
                "trajectory_path": "/trajectories/pile_simple.csv",
            },
            "Pallets Pile": {
                "usd_path": "/home/air/Downloads/Industrial_NVD@10012/Assets/ArchVis/Industrial/Piles/Pallets_A1.usd",
                "scale": [0.01, 0.01, 0.01],
                "position": [-3.0, 18.0, 0.01],
                "mass": 2.0,
                "trajectory_path": "/trajectories/testing_2.csv",
            },
        }
        # Method 1
        # def add_usd(
        #     usd_file, prim_path, pos=(0, 0, 0), orient=(1, 0, 0, 0), scale=(1, 1, 1)
        # ):
        #     orientation = (
        #         orient
        #         if len(orient) == 4
        #         else euler_angles_to_quat(orient, degrees=True)
        #     )

        #     create_prim(
        #         prim_path=prim_path,
        #         prim_type="Xform",
        #         position=pos,
        #         orientation=orientation,
        #         scale=scale,
        #     )

        #     assert os.path.exists(usd_file) == True, f"{usd_file} file Not Found"
        #     return add_reference_to_stage(os.path.abspath(usd_file), prim_path)

        # add_usd(
        #     usd_path,
        #     prim_path,
        #     pos=(-3.0, 0.5, 0.01),
        #     scale=(0.01, 0.01, 0.01),
        # )

        # Method 2
        from pxr import Usd, UsdGeom, PhysxSchema, UsdPhysics, Gf

        def add_usd_object_with_physics(
            stage,
            prim_path: str,
            usd_path: str,
            scale: list = [1.0, 1.0, 1.0],
            position: list = [0.0, 0.0, 0.0],
            orientation: list = [0.0, 0.0, 0.0, 1.0],  # Quaternion [x, y, z, w]
            mass: float = 1.0,
        ):
            # Reference the USD file to the stage
            stage.DefinePrim(prim_path, "Xform")
            prim = stage.GetPrimAtPath(prim_path)
            prim.GetReferences().AddReference(usd_path)

            # Set transform
            xform = UsdGeom.Xformable(prim)
            xform.AddTransformOp().Set(
                Gf.Matrix4d().SetScale(scale) * Gf.Matrix4d().SetTranslate(position)
            )
            # Orientation (optional, as quaternion)
            # Note: Isaac Sim often prefers transform matrix, but you can apply rotation here if needed.

            # Add collision API
            UsdPhysics.CollisionAPI.Apply(prim)

            # Add rigid body
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim)

            # Set mass using MassAPI
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr().Set(mass)

            print(f"[Success] Object added at {prim_path} with physics and collision!")

        # Example usage:
        stage = omni.usd.get_context().get_stage()
        object_des = OBJECT_ASSETS["Warehouse Pile"]
        add_usd_object_with_physics(
            stage=stage,
            prim_path=prim_path,
            usd_path=object_des["usd_path"],
            scale=object_des["scale"],
            position=object_des["position"],
            mass=object_des["mass"],
        )

        # Get the current directory used to read trajectories and save results
        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())

        # Create the vehicle 1
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor1 = MultirotorConfig()
        # Create the multirotor configuration
        mavlink_config = PX4MavlinkBackendConfig(
            {
                "vehicle_id": 0,
                "px4_autolaunch": True,
                "px4_dir": "/home/marcelo/PX4-Autopilot",
            }
        )
        config_multirotor1.backends = [
            NonlinearController(
                trajectory_file=self.curr_dir + object_des["trajectory_path"],
                # results_file=self.curr_dir + "/results/single_statistics.npz",
                Ki=[0.5, 0.5, 0.5],
                Kr=[2.0, 2.0, 2.0],
            ),
            # PX4MavlinkBackend(mavlink_config),
            # ROS2Backend(
            #     vehicle_id=1,
            #     config={
            #         "namespace": "drone",
            #         "pub_sensors": True,
            #         "pub_graphical_sensors": True,
            #         "pub_state": True,
            #         "sub_control": False,
            #     },
            # ),
        ]
        # Create a camera and lidar sensors
        # config_multirotor1.graphical_sensors = [
        #     MonocularCamera("camera", config={"update_rate": 60.0}),
        #     Lidar("lidar"),
        # ]  # Lidar("lidar")
        config_multirotor1.graphs = [
            ROS2CameraGraph(
                "body/Camera",
                config={"types": ["rgb", "camera_info", "depth_pcl", "depth"]},
            )
        ]
        Multirotor(
            "/World/quadrotor1",
            ROBOTS["Iris"],
            0,
            [-3.0, 6.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 90.0], degrees=True).as_quat(),
            config=config_multirotor1,
        )

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """
        start = time.perf_counter()
        print(f"---------------------------------------------")
        self.loop_cnt += 1
        if self.loop_cnt == self.max_loop_cnt + 1:
            # Cleanup and stop
            carb.log_warn("PegasusApp Simulation App is closing.")
            self.timeline.stop()
            simulation_app.close()

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running():

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)
            if not self.timeline.is_playing():
                end = time.perf_counter()
                elapsed_ms = (end - start) * 1000  # Chuyển giây sang mili giây
                print(f"[all] Thời gian chạy: {elapsed_ms:.3f} ms")
                self.run()


def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()


if __name__ == "__main__":
    main()
