"""
Epigent Backend - Mesa-based COVID-19 Agent-Based Model
This script provides the core simulation engine using Mesa framework
"""

import mesa
import numpy as np
import json
from enum import Enum
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass

class AgentStatus(Enum):
    HEALTHY = "healthy"
    INFECTED = "infected"
    RECOVERED = "recovered"
    VACCINATED = "vaccinated"
    DEAD = "dead"

class Environment(Enum):
    HOME = "home"
    SCHOOL = "school"
    OFFICE = "office"
    HOSPITAL = "hospital"
    RESTAURANT = "restaurant"
    OUTDOOR = "outdoor"
    MIXED = "mixed"

@dataclass
class SimulationParams:
    population_size: int = 500
    initial_infected: int = 5
    infection_probability: float = 0.1
    recovery_time: int = 14
    death_probability: float = 0.02
    vaccination_rate: float = 0.05
    environment: Environment = Environment.MIXED
    simulation_steps: int = 1000
    enable_mutation: bool = False
    social_distancing: float = 0.5
    grid_width: int = 50
    grid_height: int = 50

class CovidAgent(mesa.Agent):
    """An agent representing a person in the COVID-19 simulation"""
    
    def __init__(self, unique_id: int, model, age: int = None):
        super().__init__(unique_id, model)
        self.age = age or random.randint(0, 80)
        self.status = AgentStatus.HEALTHY
        self.infection_time = 0
        self.immunity_time = 0
        self.is_vaccinated = False
        self.social_distancing_compliance = random.random()
        
        # Age-based risk factors
        if self.age < 18:
            self.death_risk_multiplier = 0.1
        elif self.age < 65:
            self.death_risk_multiplier = 1.0
        else:
            self.death_risk_multiplier = 5.0
    
    def step(self):
        """Execute one step of agent behavior"""
        if self.status == AgentStatus.DEAD:
            return
            
        # Move agent
        self.move()
        
        # Update infection status
        if self.status == AgentStatus.INFECTED:
            self.infection_time += 1
            
            # Check for recovery or death
            if self.infection_time >= self.model.params.recovery_time:
                death_prob = (self.model.params.death_probability * 
                            self.death_risk_multiplier)
                
                if random.random() < death_prob:
                    self.status = AgentStatus.DEAD
                    self.model.stats['dead'] += 1
                    self.model.stats['infected'] -= 1
                else:
                    self.status = AgentStatus.RECOVERED
                    self.model.stats['recovered'] += 1
                    self.model.stats['infected'] -= 1
                    self.immunity_time = 365  # 1 year immunity
        
        # Check for vaccination
        if (self.status == AgentStatus.HEALTHY and 
            not self.is_vaccinated and 
            random.random() < self.model.params.vaccination_rate):
            self.is_vaccinated = True
            if random.random() < 0.95:  # 95% vaccine effectiveness
                self.status = AgentStatus.VACCINATED
                self.model.stats['vaccinated'] += 1
                self.model.stats['healthy'] -= 1
        
        # Attempt infection
        if self.status in [AgentStatus.HEALTHY, AgentStatus.VACCINATED]:
            self.check_infection()
    
    def move(self):
        """Move agent based on environment and social distancing"""
        if self.status == AgentStatus.DEAD:
            return
            
        # Reduce movement if social distancing
        if (self.social_distancing_compliance > self.model.params.social_distancing):
            if random.random() < 0.7:  # 70% chance to stay put
                return
        
        # Get possible moves
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        
        # Choose random move
        if possible_steps:
            new_position = self.random.choice(possible_steps)
            self.model.grid.move_agent(self, new_position)
    
    def check_infection(self):
        """Check if agent gets infected by nearby infected agents"""
        if self.status == AgentStatus.INFECTED or self.status == AgentStatus.DEAD:
            return
            
        # Get neighbors
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False, radius=2
        )
        
        infected_neighbors = [n for n in neighbors if n.status == AgentStatus.INFECTED]
        
        for infected_neighbor in infected_neighbors:
            # Calculate infection probability based on environment
            base_prob = self.model.params.infection_probability
            env_multiplier = self.model.get_environment_multiplier()
            
            # Reduce probability if vaccinated
            if self.status == AgentStatus.VACCINATED:
                base_prob *= 0.1  # 90% protection
            
            infection_prob = base_prob * env_multiplier
            
            if random.random() < infection_prob:
                self.status = AgentStatus.INFECTED
                self.infection_time = 0
                self.model.stats['infected'] += 1
                if self.status == AgentStatus.HEALTHY:
                    self.model.stats['healthy'] -= 1
                elif self.status == AgentStatus.VACCINATED:
                    self.model.stats['vaccinated'] -= 1
                break

class CovidModel(mesa.Model):
    """Mesa model for COVID-19 simulation"""
    
    def __init__(self, params: SimulationParams):
        super().__init__()
        self.params = params
        self.schedule = mesa.time.RandomActivation(self)
        self.grid = mesa.space.MultiGrid(
            params.grid_width, params.grid_height, torus=True
        )
        
        # Statistics tracking
        self.stats = {
            'healthy': params.population_size - params.initial_infected,
            'infected': params.initial_infected,
            'recovered': 0,
            'vaccinated': 0,
            'dead': 0
        }
        
        self.step_count = 0
        self.running = True
        
        # Data collection
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Healthy": lambda m: m.stats['healthy'],
                "Infected": lambda m: m.stats['infected'],
                "Recovered": lambda m: m.stats['recovered'],
                "Vaccinated": lambda m: m.stats['vaccinated'],
                "Dead": lambda m: m.stats['dead'],
                "R0": self.calculate_r0
            }
        )
        
        # Create agents
        self.create_agents()
        
        # Collect initial data
        self.datacollector.collect(self)
    
    def create_agents(self):
        """Create and place agents on the grid"""
        for i in range(self.params.population_size):
            agent = CovidAgent(i, self)
            
            # Set initial infection status
            if i < self.params.initial_infected:
                agent.status = AgentStatus.INFECTED
            
            # Add to scheduler
            self.schedule.add(agent)
            
            # Place on grid
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
    
    def get_environment_multiplier(self) -> float:
        """Get transmission multiplier based on environment"""
        multipliers = {
            Environment.HOME: 1.2,
            Environment.SCHOOL: 1.8,
            Environment.OFFICE: 1.4,
            Environment.HOSPITAL: 2.5,
            Environment.RESTAURANT: 2.0,
            Environment.OUTDOOR: 0.3,
            Environment.MIXED: 1.0
        }
        return multipliers.get(self.params.environment, 1.0)
    
    def calculate_r0(self) -> float:
        """Calculate effective reproduction number"""
        if self.stats['infected'] == 0:
            return 0.0
        
        # Simplified R0 calculation
        susceptible_ratio = (self.stats['healthy'] + self.stats['vaccinated'] * 0.1) / self.params.population_size
        base_r0 = 2.5  # Base R0 for COVID-19
        
        return base_r0 * susceptible_ratio * self.get_environment_multiplier()
    
    def step(self):
        """Execute one step of the model"""
        self.schedule.step()
        self.step_count += 1
        
        # Collect data
        self.datacollector.collect(self)
        
        # Check stopping conditions
        if (self.stats['infected'] == 0 or 
            self.step_count >= self.params.simulation_steps):
            self.running = False
    
    def get_agent_positions(self) -> List[Dict]:
        """Get current positions and status of all agents"""
        positions = []
        for agent in self.schedule.agents:
            if agent.pos:
                positions.append({
                    'id': agent.unique_id,
                    'x': agent.pos[0],
                    'y': agent.pos[1],
                    'status': agent.status.value,
                    'age': agent.age
                })
        return positions
    
    def get_statistics(self) -> Dict:
        """Get current simulation statistics"""
        return {
            'step': self.step_count,
            'stats': self.stats.copy(),
            'r0': self.calculate_r0(),
            'running': self.running
        }

def run_simulation(params_dict: Dict) -> Dict:
    """Run a complete simulation and return results"""
    params = SimulationParams(**params_dict)
    model = CovidModel(params)
    
    # Run simulation
    while model.running:
        model.step()
    
    # Get results
    model_data = model.datacollector.get_model_vars_dataframe()
    
    return {
        'steps': len(model_data),
        'final_stats': model.stats,
        'time_series': model_data.to_dict('records'),
        'peak_infections': model_data['Infected'].max(),
        'total_deaths': model.stats['dead'],
        'final_r0': model.calculate_r0()
    }

def run_batch_simulations(params_dict: Dict, num_runs: int = 10) -> Dict:
    """Run multiple simulations and return aggregated results"""
    results = []
    
    for run in range(num_runs):
        print(f"Running simulation {run + 1}/{num_runs}")
        result = run_simulation(params_dict)
        results.append(result)
    
    # Aggregate results
    peak_infections = [r['peak_infections'] for r in results]
    total_deaths = [r['total_deaths'] for r in results]
    
    return {
        'num_runs': num_runs,
        'peak_infections': {
            'mean': np.mean(peak_infections),
            'std': np.std(peak_infections),
            'min': np.min(peak_infections),
            'max': np.max(peak_infections)
        },
        'total_deaths': {
            'mean': np.mean(total_deaths),
            'std': np.std(total_deaths),
            'min': np.min(total_deaths),
            'max': np.max(total_deaths)
        },
        'individual_results': results
    }

if __name__ == "__main__":
    # Example usage
    params = {
        'population_size': 500,
        'initial_infected': 5,
        'infection_probability': 0.1,
        'recovery_time': 14,
        'death_probability': 0.02,
        'vaccination_rate': 0.05,
        'environment': 'mixed',
        'simulation_steps': 400,
        'enable_mutation': False,
        'social_distancing': 0.5
    }
    
    # Run single simulation
    result = run_simulation(params)
    print("Simulation completed!")
    print(f"Peak infections: {result['peak_infections']}")
    print(f"Total deaths: {result['total_deaths']}")
    print(f"Final R0: {result['final_r0']:.2f}")
