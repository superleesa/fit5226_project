import unittest
from fit5226_project.env import Assignment2Environment, Action, State, Assignment2State

class TestAssignment2Environment(unittest.TestCase):

    def setUp(self):
        self.env = Assignment2Environment(n=5, with_animation=False)
        self.env.initialize_for_new_episode(agent_location=(0, 0))

    def test_initialize_for_new_episode(self):
        self.env.initialize_for_new_episode(agent_location=(1, 1))
        state = self.env.get_state()
        print(state.agent_location, state.item_location, state.goal_location)
        self.assertEqual(state.agent_location, (1, 1))
        self.assertFalse(state.has_item)

    def test_get_available_actions(self):
        actions = self.env.get_available_actions()
        self.assertIn(Action.RIGHT, actions)
        self.assertIn(Action.UP, actions)
        self.assertNotIn(Action.LEFT, actions)
        self.assertNotIn(Action.DOWN, actions)

    def test_get_reward(self):
        prev_state = self.env.get_state()
        self.env.update_state(Action.RIGHT)
        current_state = self.env.get_state()
        reward = self.env.get_reward(prev_state, current_state, Action.RIGHT)
        self.assertIsInstance(reward, float)

    def test_step(self):
        reward, next_state = self.env.step(Action.RIGHT)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(next_state, Assignment2State)
        self.assertEqual(next_state.agent_location, (1, 0))

    def test_is_goal_state(self):
        state = self.env.get_state()
        self.assertFalse(self.env.is_goal_state(state))
        # Manually set the state to goal state for testing
        self.env.current_sub_environment.state.has_item = True
        self.env.current_sub_environment.goal_location = state.agent_location
        self.assertTrue(self.env.is_goal_state(state))


if __name__ == '__main__':
    unittest.main()