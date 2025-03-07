from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs import policy


class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)
        self.rotation_constraint = getattr(config, section).rotation_constraint
        self.kinematics = getattr(config, section).kinematics

    def act(self, ob, supervised=False):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        if hasattr(self.policy, 'run_solver'):
            action, action_index = self.policy.predict(state, supervised)
        else:
            action, action_index = self.policy.predict(state)

        return action, action_index

    def get_state(self, ob, as_tensors=True):
        if as_tensors:
            if self.policy is None:
                raise AttributeError('Policy attribute has to be set!')
            return self.policy.transform(JointState(self.get_full_state(), ob))
        else:
            return JointState(self.get_full_state(), ob)
