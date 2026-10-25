"""RoboDojo control acceptance: both ARX X5 arms, a gripper cycle and an RGB head-camera frame."""
from rosbridge_runtime import dual_arm_demo


def main():
    dual_arm_demo.main("robodojo_control_demo", "head_camera", depth=False)


if __name__ == "__main__":
    main()
