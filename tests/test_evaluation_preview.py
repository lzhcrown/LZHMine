"""Viewer pause/focus and preview lifecycle must not create captures."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import isaacgym
import pytest
import torch

from legged_gym.evaluation.preview import PreviewViewer, run_preview


def test_viewer_focus_and_pause_process_events_without_stepping_physics(monkeypatch):
    monkeypatch.setattr('legged_gym.evaluation.preview.time.sleep', lambda seconds: None)
    gym = NS(subscribe_viewer_keyboard_event=Mock(), query_viewer_has_closed=Mock(return_value=False),
             query_viewer_action_events=Mock(side_effect=[
                 [NS(action='next_robot', value=1), NS(action='pause_preview', value=1)],
                 [NS(action='previous_robot', value=1), NS(action='pause_preview', value=1)],
                 [NS(action='QUIT', value=1)]]), step_graphics=Mock(), draw_viewer=Mock(), simulate=Mock())
    env = NS(gym=gym, viewer=object(), sim=object(), device='cpu', num_envs=2,
             terrain=NS(type_names=['flat', 'platform_45cm']), terrain_types=[0, 1],
             root_states=torch.tensor([[1., 2., 3.], [4., 5., 6.]]), set_camera=Mock())
    viewer = PreviewViewer(env)
    viewer.render()
    assert viewer.focus == 0 and not viewer.paused
    assert gym.draw_viewer.call_count == 2
    gym.simulate.assert_not_called()
    assert env.set_camera.call_args_list[0].args[1] == [4., 5., 6.]
    assert env.set_camera.call_args_list[1].args[1] == [1., 2., 3.]
    with pytest.raises(KeyboardInterrupt):
        viewer.render()


def test_headless_preview_uses_no_sinks_and_always_destroys_sim(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    env = NS(viewer=None, sim=object(), num_obs=2, dt=.02, evaluation_complete=False,
             completed_episodes=[], gym=NS(destroy_sim=Mock()), reset=Mock(),
             get_observations=Mock(return_value=torch.zeros(1, 2)))
    def step(actions):
        env.evaluation_complete = True
        env.completed_episodes.append(dict(env_id=0, episode_index=1, trial_elapsed_s=30.,
                                          terrain='platform_45cm', status='timeout', reason='time_limit'))
    env.step = step
    factory = Mock(return_value=env)
    args = NS(sim_device='cpu', viewer=False, no_record=True)
    policy = NS(num_actor_obs=2, act_inference=lambda obs: torch.zeros(1, 16))
    assert run_preview(args, object(), object(), object(), policy, factory) == 0
    assert 'env 0 | platform_45cm | round 2 | 30.00 s | timeout (time_limit)' in capsys.readouterr().out
    assert factory.call_args.kwargs == {}
    env.gym.destroy_sim.assert_called_once_with(env.sim)
    assert list(tmp_path.iterdir()) == []
