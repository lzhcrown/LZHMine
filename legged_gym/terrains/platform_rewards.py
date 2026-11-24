"""High-platform reward settings for Nezha's mixed-terrain task."""


def platform_reward_scales():
    """Per-second coefficients, converted to control-step units by the environment.

    MUJICA tracks planar speed and regularizes vertical speed, roll and yaw
    rates during climbing. The other terms and magnitudes follow Dreamwaq M20.
    """
    return {
        "tracking_lin_vel": 3.0,
        "lin_vel_z": -2.0,
        "ang_vel_xz": -0.05,
        "torques": -5e-6,
        "dof_vel": -1e-6,
        "dof_acc": -1e-7,
        "collision": -1.0,
        "action_rate": -0.011,
        "stand_still": -0.5,
        "dof_pos_limits": -5.0,
        "hip_default": -0.5,
        "run_still": -0.05,
        "termination": -0.8,
    }


def platform_v1_reward_scales():
    """Dreamwaq M20 rewards with MUJICA's vertical speed and roll/yaw rate costs."""
    return dict(
        platform_reward_scales(),
        tracking_ang_vel=1.5,
        orientation=-0.5,
        base_height=-10.0,
    )


def platform_climb_settings():
    """G13 event rewards are per achievement, independent of the control period."""
    return {
        "upper_support_reward": 4.0,
        "completion_reward": 6.0,
        "settle_time_s": 0.20,
    }
