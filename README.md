# rocketlander
Rocket Landing in a Gimbal-Based Environment using Reinforcement Learning Algorithms

### Action Space
A list consisting of 2 elements:
- Thrust: a value between 0.0 and 2.0, represents the G's of thrust
- Gimbal: a value between -30 and 30, representing the nozzle angular velocity

### State Space
The state of the rocket environment is represented by an 7-dimensional vector, capturing essential details for controlling the rocket. Each component of the state vector is normalized:
- x: Horizontal position (m)
- y: Vertical position (m)
- vx: Horizontal velocity (m/s)
- vy: Vertical velocity (m/s)
- θ (theta): Rocket’s angle relative to the vertical (radians)
- vθ (vtheta): Angular velocity of the rocket (radians/s)
- φ (phi): Nozzle angle (radians)

### Reward Structure
???