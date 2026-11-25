"""Interactive inspection of the same evaluation environment, without files."""
import time


class PreviewViewer:
    """Follow one robot and switch among terrain slots without changing physics."""

    def __init__(self, env):
        from isaacgym import gymapi
        self.env = env
        self.focus = 0
        self.paused = False
        for key, action in ((gymapi.KEY_N, 'next_robot'), (gymapi.KEY_B, 'previous_robot'),
                            (gymapi.KEY_SPACE, 'pause_preview')):
            env.gym.subscribe_viewer_keyboard_event(env.viewer, key, action)
        self.describe()

    def describe(self):
        env = self.env
        name = env.terrain.type_names[int(env.terrain_types[self.focus])]
        print('Following env {} / {} | N next, B previous, Space pause/resume, Esc stop'.format(
            self.focus, name), flush=True)

    def render(self, sync_frame_time=True):
        env = self.env
        # Called by env.step before integration. Pausing stays in this loop, so
        # neither simulation time nor the attempt timeout advances.
        while True:
            if env.gym.query_viewer_has_closed(env.viewer):
                raise KeyboardInterrupt
            for event in env.gym.query_viewer_action_events(env.viewer):
                if event.value <= 0:
                    continue
                if event.action == 'QUIT':
                    raise KeyboardInterrupt
                if event.action == 'pause_preview':
                    self.paused = not self.paused
                    print('Paused' if self.paused else 'Resumed', flush=True)
                if event.action in ('next_robot', 'previous_robot'):
                    delta = 1 if event.action == 'next_robot' else -1
                    self.focus = (self.focus + delta) % env.num_envs
                    self.describe()
            if env.device != 'cpu':
                env.gym.fetch_results(env.sim, True)
            target = env.root_states[self.focus, :3].detach().cpu().tolist()
            env.set_camera([target[0] - 3.5, target[1] - 4., target[2] + 2.5], target)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            if not self.paused:
                return
            time.sleep(.02)


def run_preview(args, cfg, sim_params, physics_engine, policy, environment_class):
    import torch
    env = None
    status = 'error'
    try:
        env = environment_class(cfg, sim_params, physics_engine, args.sim_device, not args.viewer)
        if policy.num_actor_obs != env.num_obs:
            raise ValueError('Checkpoint observation size differs from environment')
        env.reset()
        if args.viewer:
            viewer = PreviewViewer(env)
            env.render = viewer.render
        print('Preview only: no evaluation files will be created.', flush=True)
        with torch.inference_mode():
            while not env.evaluation_complete:
                start = time.monotonic()
                env.step(policy.act_inference(env.get_observations()))
                for row in env.completed_episodes:
                    print('env {env_id} | {terrain} | round {number} | {trial_elapsed_s:.2f} s | '
                          '{status} ({reason})'.format(number=row['episode_index'] + 1, **row), flush=True)
                env.completed_episodes.clear()
                if args.viewer:
                    # Gym's frame clock is the physics dt, while this loop runs
                    # at the slower control rate. Pace by the actual control dt.
                    time.sleep(max(0., env.dt - (time.monotonic() - start)))
        status = 'complete'
    except (KeyboardInterrupt, SystemExit):
        status = 'interrupted'
    finally:
        if env is not None:
            if env.viewer is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)
    print('Preview {}. No evaluation files saved.'.format(status), flush=True)
    return 0 if status == 'complete' else 130
