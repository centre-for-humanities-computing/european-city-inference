import unittest
import numpy as np
from multi_agent_simulation import MultiAgentSimulation  # Supposons que le code est dans un fichier multi_agent_simulation.py

class TestMultiAgentSimulation(unittest.TestCase):
    def setUp(self):
        # Configuration commune pour tous les tests
        self.n_steps = 10
        self.n_nodes = 2
        self.preference = np.array([0.3, 0.4, 0.3])
        self.n_agents = 3
        self.simulation = MultiAgentSimulation(n_steps=self.n_steps, n_nodes=self.n_nodes, preference=self.preference)

    def test_initialization(self):
        """Test that the MultiAgentSimulation is initialized correctly."""
        self.assertEqual(self.simulation.n_steps, self.n_steps)
        self.assertEqual(self.simulation.n_nodes, self.n_nodes)
        np.testing.assert_array_equal(self.simulation.preference, self.preference)
        self.assertEqual(len(self.simulation.agents), 0)
        self.assertEqual(len(self.simulation.observations), 0)

    def test_create_agent(self):
        """Test that the create_agent method creates an agent with the correct structure."""
        agent = self.simulation.create_agent()
        # Vérifier que l'agent est créé
        self.assertIsNotNone(agent)
        # Vérifier que l'agent a le bon nombre de nœuds (approximatif)
        # Note: Cela dépend de l'implémentation de Network dans pyhgf
        # Pour une vérification plus précise, nous aurions besoin de connaître la structure interne de Network
        self.assertTrue(hasattr(agent, 'nodes'))

    def test_generate_observations(self):
        """Test that the generate_observations method generates observations with the correct shape."""
        self.simulation.generate_observations(self.n_agents)
        self.assertEqual(len(self.simulation.observations), self.n_agents)
        for obs in self.simulation.observations:
            self.assertEqual(obs.shape, (self.n_steps, self.n_nodes))
            # Vérifier que les valeurs sont dans la plage attendue pour une distribution bêta(1, 0.2)
            self.assertTrue(np.all(obs >= 0) and np.all(obs <= 1))

    def test_run_simulation(self):
        """Test that the run_simulation method creates the correct number of agents."""
        self.simulation.run_simulation(self.n_agents)
        self.assertEqual(len(self.simulation.agents), self.n_agents)
        self.assertEqual(len(self.simulation.observations), self.n_agents)

    def test_calculate_surprise(self):
        """Test that the calculate_surprise method returns a valid value."""
        expected_mean = 0.5
        actual_value = 0.7
        surprise = self.simulation.calculate_surprise(expected_mean, actual_value)
        self.assertIsInstance(surprise, float)
        # Vérifier que la surprise est non négative
        self.assertGreaterEqual(surprise, 0)

        # Tester avec des valeurs égales (la surprise devrait être minimale)
        equal_surprise = self.simulation.calculate_surprise(0.5, 0.5)
        self.assertGreaterEqual(equal_surprise, 0)
        self.assertLess(equal_surprise, surprise)  # La surprise devrait être plus faible lorsque les valeurs sont proches

    def test_calculate_kl_divergence(self):
        """Test that the calculate_kl_divergence method returns a valid value."""
        # Créer des distributions de test (simplifiées)
        # Note: dirichlet_kullback_leibler attend probablement des paramètres de distribution Dirichlet
        # Pour cet exemple, nous utilisons des tableaux simples
        distribution1 = np.array([0.5, 0.5])
        distribution2 = np.array([0.3, 0.7])
        kl_divergence = self.simulation.calculate_kl_divergence(distribution1, distribution2)
        self.assertIsInstance(kl_divergence, float)
        # Vérifier que la divergence de KL est non négative
        self.assertGreaterEqual(kl_divergence, 0)

        # Tester avec des distributions identiques (la divergence devrait être 0)
        identical_kl = self.simulation.calculate_kl_divergence(distribution1, distribution1)
        self.assertAlmostEqual(identical_kl, 0)

    def test_get_agent_surprises(self):
        """Test that the get_agent_surprises method returns a matrix of the correct shape."""
        self.simulation.run_simulation(self.n_agents)
        time_step = 0  # Premier pas de temps
        surprises = self.simulation.get_agent_surprises(time_step)
        self.assertEqual(surprises.shape, (self.n_agents, self.n_nodes))
        # Vérifier que toutes les valeurs sont non négatives
        self.assertTrue(np.all(surprises >= 0))

    def test_get_agent_kl_divergences(self):
        """Test that the get_agent_kl_divergences method returns a tensor of the correct shape."""
        self.simulation.run_simulation(self.n_agents)
        time_step = 0  # Premier pas de temps
        kl_divergences = self.simulation.get_agent_kl_divergences(time_step)
        self.assertEqual(kl_divergences.shape, (self.n_agents, self.n_agents, self.n_nodes))
        # Vérifier que toutes les valeurs sont non négatives (en ignorant les NaN)
        non_nan_values = kl_divergences[~np.isnan(kl_divergences)]
        if len(non_nan_values) > 0:
            self.assertTrue(np.all(non_nan_values >= 0))

    def test_plot_trajectories(self):
        """Test that the plot_trajectories method runs without errors."""
        # Ce test vérifie simplement que la méthode s'exécute sans erreur
        # Il ne vérifie pas le contenu du graphique (difficile à automatiser)
        self.simulation.run_simulation(self.n_agents)
        try:
            self.simulation.plot_trajectories()
        except Exception as e:
            self.fail(f"plot_trajectories raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()
