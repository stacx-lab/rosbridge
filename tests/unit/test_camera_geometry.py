import numpy as np
import pytest

from rosbridge.catalog.cameras import color, distort, undistort
from rosbridge.core.lens import LIMIT, LensGeometry

# HandsOn's head camera: the Gemini 336's calibrated lens (k1, k2, p1, p2, k3).
LENS = (0.00646656053141, -0.0479530096054, -0.000352804054273, -0.000104829683551, 0.0332237109542)


def test_undistort_inverts_the_lens_across_the_image():
    x, y = np.meshgrid(np.linspace(-1., 1., 21), np.linspace(-.6, .6, 13))
    np.testing.assert_allclose(distort(*undistort(x, y, LENS), LENS), (x, y), atol=1e-12)
    assert np.abs(np.subtract(undistort(x, y, LENS), (x, y))).max() > 1e-3


def test_color_states_a_lens_and_its_field_of_view_or_an_undistorted_pinhole():
    pinhole = color(854, 480, 463., 463., 426.5, 239.5)
    assert pinhole["distortion"] == {"model": "none"} and "fov" not in pinhole and pinhole["fps"] is None
    lensed = color(854, 480, 463., 463., 426.5, 239.5, LENS)
    assert lensed["distortion"].pop("model") == "opencv_brown_conrady"
    assert lensed["distortion"] == pytest.approx(dict(zip(("k1", "k2", "p1", "p2", "k3"), LENS)))
    # A zero lens has the pinhole's field of view; this lens widens it.
    flat = color(854, 480, 463., 463., 426.5, 239.5, (0.,) * 5)["fov"]
    assert flat["horizontal_deg"] == pytest.approx(2 * np.degrees(np.arctan(427 / 463.)))
    assert flat["vertical_deg"] == pytest.approx(2 * np.degrees(np.arctan(240 / 463.)))
    assert lensed["fov"]["horizontal_deg"] > flat["horizontal_deg"]


def head_camera():
    return LensGeometry("head_camera/base", 854, 480, 463.393371582031, 463.001892089844, 431.267272949219,
                        239.418975830078, LENS)


def test_geometry_describes_the_color_stream_as_its_profile_states_it():
    lens = head_camera()
    stream = lens.geometry()["streams"]["color"]
    stated = color(854, 480, 463.393371582031, 463.001892089844, 431.267272949219, 239.418975830078, LENS)
    assert {key: stream[key] for key in ("width", "height", "projection", "distortion")} == {
        key: stated[key] for key in ("width", "height", "projection", "distortion")}
    assert stream["frame"] == "head_camera/base" and len(stream["calibration_id"]) == 32
    assert set(stream) == {"width", "height", "projection", "distortion", "frame", "calibration_id"}
    stream["width"] = 0  # a reply is the caller's own copy
    assert lens.geometry()["streams"]["color"]["width"] == 854
    assert head_camera().descriptor["calibration_id"] != stream["calibration_id"]  # one per loaded camera


def test_pixels_and_rays_convert_through_the_lens_and_back():
    lens = head_camera()
    calibration = lens.descriptor["calibration_id"]
    pixels = [[431.267272949219, 239.418975830078], [0., 0.], [853., 479.], [100.5, 400.25]]
    rays = lens.unproject("color", calibration, pixels)["rays"]
    assert rays[0] == pytest.approx([0., 0., 1.], abs=1e-12) and all(ray[2] == 1. for ray in rays)
    # The corner's ray is not the pinhole's: the lens moves it by pixels.
    pinhole = [(0. - 431.267272949219) / 463.393371582031, (0. - 239.418975830078) / 463.001892089844]
    assert np.hypot(*np.subtract(rays[1][:2], pinhole)) * 463. > 1.
    points = [[3. * x, 3. * y, 3.] for x, y, _ in rays]  # any depth along the ray
    np.testing.assert_allclose(lens.project("color", calibration, points)["pixels"], pixels, atol=1e-8)
    cv2 = pytest.importorskip("cv2")
    k = np.array([[463.393371582031, 0, 431.267272949219], [0, 463.001892089844, 239.418975830078], [0, 0, 1]])
    expected, _ = cv2.projectPoints(np.array(points).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, np.array(LENS))
    np.testing.assert_allclose(lens.project("color", calibration, points)["pixels"], expected.reshape(-1, 2), atol=1e-6)


def test_invalid_entries_come_back_null_and_invalid_requests_are_refused():
    lens = head_camera()
    calibration = lens.descriptor["calibration_id"]
    assert lens.unproject("color", calibration, [])["rays"] == []
    rays = lens.unproject("color", calibration, [[10., 20.], [float("nan"), 0.], [float("inf"), 1.]])["rays"]
    assert rays[0] is not None and rays[1:] == [None, None]
    pixels = lens.project("color", calibration, [[0., 0., 1.], [0., 0., 0.], [1., 1., -2.], [0., float("nan"), 1.]])
    assert pixels["pixels"][0] == pytest.approx([431.267272949219, 239.418975830078]) and pixels["pixels"][1:] == [None] * 3
    with pytest.raises(ValueError, match="stale or unknown camera calibration; refresh geometry"):
        lens.unproject("color", "0" * 32, [[1., 2.]])
    with pytest.raises(ValueError, match="stale or unknown camera calibration"):
        lens.unproject("depth", calibration, [[1., 2.]])
    with pytest.raises(ValueError, match="invalid coordinate shape"):
        lens.unproject("color", calibration, [[1., 2., 3.]])
    with pytest.raises(ValueError, match="invalid coordinate shape"):
        lens.project("color", calibration, [[1., 2.]])
    with pytest.raises(ValueError, match="coordinates must be numbers"):
        lens.unproject("color", calibration, [[True, 2.]])
    with pytest.raises(ValueError, match="coordinates must be numbers"):
        lens.unproject("color", calibration, [["1", 2.]])
    with pytest.raises(ValueError, match=f"at most {LIMIT} coordinates"):
        lens.unproject("color", calibration, [[1., 2.]] * (LIMIT + 1))
    with pytest.raises(ValueError, match=f"at most {LIMIT} coordinates"):
        lens.unproject("color", calibration, {"u": 1})
    assert len(lens.unproject("color", calibration, [[1., 2.]] * LIMIT)["rays"]) == LIMIT
