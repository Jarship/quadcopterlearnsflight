import numpy as np
from physics_sim import PhysicsSim
from task import Task

class Soft_Landing_Task(Task):
    ''' Task environment for a soft landing, that gives feedback, as defined by John Bishop'''
    def __init__(self, init_pose = None, init_velocities = None, 
                 init_angle_velocities = None, runtime = 5, target_pos = None):
        """ Initialize the soft_landing_task object
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocities of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: Time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """

        self.given = init_pose is not None
        self.init_pose = init_pose if self.given else self.generate_values(1)
        #Goal
        self.target_pos = target_pos if target_pos is not None else np.array([self.init_pose[0], self.init_pose[1], 0])
        
        super().__init__(self.init_pose,init_velocities,init_angle_velocities,runtime,self.target_pos)
        self.state_size = self.action_repeat * 7
        
    def get_reward(self):
        """Uses current position, Euler Angles, and Velocity of sim to return reward"""
        distance = self.sim.pose[2] / 300.0 
        cross_over = 1.0/300.0 #The point we are trying to have the quadcopter go to
        if distance > cross_over:
            dist_reward = 1 - (distance - cross_over)**.4 #The reward value that scales with distance
            reward = dist_reward
                
        else:
            vel_disc = 1 / ((self.sim.v[2])**2 + 1) #Inverse squared velocity scales the velocity incentive to 1 at the highest
            if distance == 0.0:
                if self.sim.v[2] >=-1.0:
                    reward = 150 #As long as the quadcopter isn't flying down too fast, then you've landed
                else:    
                    reward = 5
            else:
                
                reward = 4 * vel_disc #Incentivize a low velocity
        return reward
    def generate_values(self, val):
        """ Generate values to randomize the starting values
        Params
        ======
        val: 1 for init_pose, 2 for init_velocities, 3 for init_angle_velocities"""
        if val == 1:
            self.generated = 0
            gen_x = (np.random.ranf() -.5) * 300  #Range between -150 to 150 
            gen_y = (np.random.ranf()-.5) * 300  #Range between -150 to 150
            gen_z = np.random.ranf() * 300      #Range between 0 and 300
        
            gen_roll = (np.random.ranf() - .5) * 2 * np.pi   #Range between -pi and pi
            gen_yaw = (np.random.ranf() - .5) * 2 * np.pi    #Range between -pi and pi
            gen_pitch = (np.random.ranf() - .5) * 2 * np.pi  #Range between -pi and pi
                         
            return np.array([gen_x,gen_y,gen_z,gen_roll,gen_yaw,gen_pitch])
        elif val == 2:
            gen_v1 = (np.random.ranf() - .5) * 600 #Range of -300 to 300
            gen_v2 = (np.random.ranf() - .5) * 600 #Range of -300 to 300
            gen_v3 = (np.random.ranf() - .5) * 600 #Range of -300 to 300
            return np.array([gen_v1, gen_v2, gen_v3])
        elif val == 3:
            angen_v1 = (np.random.ranf() - .5) * 4 * np.pi #Range of -2pi to 2pi
            angen_v2 = (np.random.ranf() - .5) * 4 * np.pi #Range of -2pi to 2pi
            angen_v3 = (np.random.ranf() - .5) * 4 * np.pi #Range of -2pi to 2pi
            return np.array([angen_v1, angen_v2, angen_v3])
        else:
            return None
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds)# update the sim pose and velocities
            reward += self.get_reward() 
            self.prev_vel = self.sim.v[2]
            pose_all.append([*self.sim.pose,self.sim.v[2]])
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    def switch(self, runtime = 5):
        """Set the sim to start generating init_pose every 25 episodes"""
        self.given = False
        self.init_pose = self.generate_values(1)
        self.sim.init_pose = self.init_pose
        self.sim.runtime = runtime
    def reset(self):
        """Reset the sim to start a new episode."""
        if not self.given:
            if self.generated > 24: 
                self.switch()
            else:
                self.generated += 1
        self.sim.reset()
        state = np.concatenate([[*self.sim.pose, self.sim.v[2]]] * self.action_repeat) 
        return state