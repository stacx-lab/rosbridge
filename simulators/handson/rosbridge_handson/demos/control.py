"""HandsOn control acceptance: both ROKAE arms, a right-hand curl and an RGB head-camera frame."""
from rosbridge_runtime import dual_arm_demo

# Index-finger MCP flexion, PIP and DIP (finger2 joints 1, 3 and 4) in the hand's finger-major order.
CURL = {4: .6, 6: .6, 7: .6}


def main():
    # joint1 swings a hand vertically and the arms mirror each other: these swings lift both hands.
    dual_arm_demo.main("handson_control_demo", "head_camera", depth=False, hand=CURL, offsets=(-.2, .15))


if __name__ == "__main__":
    main()
