import pytest


@pytest.fixture(scope="session")
def can_render():
    """Offscreen rendering needs OpenGL; headless CI runners may not have it."""
    try:
        from robot_arm_il.env import PickPlaceEnv

        env = PickPlaceEnv()
        env.reset(seed=0)
        env.render("front")
        env.close()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no offscreen OpenGL: {exc}")
