"""
Petla nadzorcy 20 Hz (docs 5.1) — spina wszystkie klocki
========================================================
raz na epizod: pomiar pozy boardu (AprilTag) -> geometria celow z CAD
co cykl:      stan robota -> detektor wcisniecia -> maszyna stanow ->
              obs (bit w bit) -> polityka (lub retry-odjazd) ->
              warstwa bezpieczenstwa -> komenda kartezjanska
Polityka odpowiada WYLACZNIE za ruch; logika zadania i bezpieczenstwo
sa deterministyczne i audytowalne.
"""

from .state_machine import POLICY, RETREAT


class Supervisor:

    def __init__(self, io, perception, detector, task_sm, retry, safety,
                 obs_builder, policy_fn):
        """policy_fn: flat_obs -> akcja (>=6 wymiarow; uzywamy [:6])."""
        self.io = io
        self.perception = perception
        self.detector = detector
        self.task_sm = task_sm
        self.retry = retry
        self.safety = safety
        self.obs_builder = obs_builder
        self.policy_fn = policy_fn

    def run_episode(self, order=None, max_steps=1000) -> dict:
        state = self.io.reset()
        if order is None:  # SimIO: specyfikacja zadania znana po resecie env
            order = self.io.task_order()
        self.perception.reset(self.io.measure_board_pose())
        self.task_sm.reset(order)
        self.detector.reset()
        self.retry.reset()
        self.safety.reset()

        steps = presses_detected = retreat_steps = 0
        while (steps < max_steps and self.io.episode_running()
               and not self.task_sm.done):
            tgt = self.task_sm.current_target()
            btn_top = self.perception.btn_top(tgt)
            axis = self.perception.press_axis(tgt)

            mode = self.retry.update(state.tcp_pos, btn_top, axis)
            if mode == RETREAT:
                action = self.retry.retreat_action(state.tcp_pos)
                retreat_steps += 1
            else:
                obj = self.obs_builder.object_obs(
                    self.perception, self.task_sm, state.tcp_pos,
                    state.tcp_quat)
                obs = self.obs_builder.build(state.proprio, obj)
                action = self.policy_fn(obs)[:6]

            action = self.safety.filter(action, state.tcp_pos,
                                        state.wrench_force)
            state = self.io.apply(action)
            steps += 1

            if self.detector.update(state.tcp_pos, btn_top, axis,
                                    state.wrench_force):
                presses_detected += 1
                self.task_sm.on_press()
                self.detector.reset()
                self.retry.on_progress()

        return {
            "steps": steps,
            "presses_detected": presses_detected,
            "sm_done": self.task_sm.done,
            "retries": self.retry.retries,
            "retreat_steps": retreat_steps,
            "force_stops": self.safety.force_stops,
        }
