"""RoboTwin control acceptance: both ARX5 arms, a gripper cycle and an RGB-D head-camera capture."""
from rosbridge_runtime import dual_arm_demo


def main():
    dual_arm_demo.main("robotwin_control_demo", "head_camera")


if __name__ == "__main__":
    main()
