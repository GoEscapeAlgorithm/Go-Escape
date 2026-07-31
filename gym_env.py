import os
from typing import Optional
import numpy as np
import gymnasium as gym
import numpy as np
import pymunk
import pygame
import env_functions as fcs
import math
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.env_util import make_vec_env

class GoEscapeEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None, ray_num: int = 24):
        self.render_mode = render_mode
        self.ray_num = ray_num
        metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 60,
        }
        # General settings
        self.SCREEN_DIMENSIONS = (300, 600)
        self.GRAVITY = 450

        # Collision Types
        BALL_TYPE = 0
        STATIC_TERRAIN_TYPE = 1
        SPIKE_TYPE = 2
        GOAL_TYPE = 3
        CONVEYOR_TYPE = 4
        ARC_TYPE = 5
        HINGE_TYPE = 6

        # Pygame and pymunk initialization
        pygame.init()
        if self.render_mode == 'human':
            self.screen = pygame.display.set_mode(self.SCREEN_DIMENSIONS)
            self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface(self.SCREEN_DIMENSIONS)
        self.space = pymunk.Space()
        self.space.gravity = 0.0, -self.GRAVITY

        # Level and state storage
        self.win_state = 0
        self.can_jump = False
        self.object_visit_order = []
        self.platforms = []
        self.spikes = []
        self.conveyors = []
        self.goal = []
        self.arcs = []
        self.hinges = []
        self.start = []
        self.num_frames_passed = 0

        # World border initialization
        self.border_body = self.space.static_body
        self.bottom_shape = pymunk.Segment(self.border_body, (-60, -60), (self.SCREEN_DIMENSIONS[0] * 2, -60), radius=10)
        self.left_shape = pymunk.Segment(self.border_body, (-60, -60), (-60, self.SCREEN_DIMENSIONS[1] * 2), radius=10)
        self.right_shape = pymunk.Segment(self.border_body, (self.SCREEN_DIMENSIONS[0] + 60, -60), (self.SCREEN_DIMENSIONS[0] + 60, self.SCREEN_DIMENSIONS[1] * 2), radius=10)
        self.bottom_shape.collision_type = SPIKE_TYPE
        self.space.add(self.bottom_shape)
        self.left_shape.collision_type = SPIKE_TYPE
        self.space.add(self.left_shape)
        self.right_shape.collision_type = SPIKE_TYPE
        self.space.add(self.right_shape)

        # Level Creation - use extend instead of append
        self.start.extend(fcs.add_start(self.space, 150))
        self.platforms.extend(fcs.add_platform(self.space, 150, 500, 0.25))
        self.spikes.extend(fcs.add_platform_spike(self.space, self.platforms[0][1], num_spikes=2))
        self.conveyors.extend(fcs.add_conveyor(self.space, 200, 150, 100, 140))
        self.spikes.extend(fcs.add_conveyor_spike(self.space, self.conveyors[0][1], left=False, num_spikes=2))
        self.arcs.extend(fcs.add_arc(self.space, 175, 250, 50, +0.5236, -0.5236))
        self.spikes.extend(fcs.add_arc_spike(self.space, self.arcs[0][0], num_spikes=2))
        self.spikes.extend(fcs.add_arc_spike(self.space, self.arcs[0][0], num_spikes=2, from_start=True, outside=False, offset=25))
        self.goal.extend(fcs.add_goal(self.space, 50, 50))
        self.platforms.extend(fcs.add_platform(self.space, 73, 70, -0.5, width=90))
        self.hinges.extend(fcs.add_hinge(self.space, 0, 400, 100, -0.7854, centered=False, left=True))
        self.spikes.extend(fcs.add_hinge_spike(self.space, self.hinges[0][1], num_spikes=2))

        # Ball initialization
        self.ball_body = pymunk.Body(mass = 10, moment = 1, body_type=pymunk.Body.DYNAMIC)
        self.ball_body.position = (self.start[0].position.x + 7, self.start[0].position.y + 13)
        self.ball_body.radius = 10
        self.ball_body.data = self.ball_body.mass, self.ball_body.moment
        self.ball_shape = pymunk.Circle(self.ball_body, 10)
        self.ball_shape.collision_type = BALL_TYPE
        self.ball_shape.filter = pymunk.ShapeFilter(categories=0x1)
        self.space.add(self.ball_body, self.ball_shape)

        
        self.space.on_collision(BALL_TYPE, STATIC_TERRAIN_TYPE, begin=self.begin_platform, pre_solve=self.set_jump_true, separate=self.set_jump_false)
        self.space.on_collision(BALL_TYPE, CONVEYOR_TYPE, begin=self.move_conveyor, pre_solve=self.check_conveyor_state, separate=self.stop_movement)
        self.space.on_collision(BALL_TYPE, SPIKE_TYPE, begin=self.end_fail)
        self.space.on_collision(BALL_TYPE, GOAL_TYPE, begin=self.end_win)
        self.space.on_collision(BALL_TYPE, ARC_TYPE, pre_solve=self.check_arc_state, separate=self.separate_arc)
        self.space.on_collision(BALL_TYPE, HINGE_TYPE, pre_solve=self.move_hinge, separate=self.set_jump_false)

        input_amount = (self.ray_num * 2) + 5
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(input_amount,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

    def _get_obs(self):
        stuffnthings = [self.ball_body.position.x, self.ball_body.position.y, self.ball_body.velocity.x, self.ball_body.velocity.y, float(self.can_jump)]
        
        ray_range = 300.0
        ray_scans = []
        angle_diff = (2 * math.pi) / self.ray_num

        for fah in range(self.ray_num):
            curr_angle = fah * angle_diff
            ray_direction = pymunk.Vec2d(math.cos(curr_angle), math.sin(curr_angle))
            ray_tar = self.ball_body.position + (ray_direction * ray_range)
            
            ray_filter = pymunk.ShapeFilter(categories=0xFFFFFFFE)
            hit = self.space.segment_query_first(self.ball_body.position, ray_tar, radius=0, shape_filter=ray_filter)

            if hit and hit.shape:
                distance = self.ball_body.position.get_distance(hit.point) / ray_range
                arc_speed = hit.shape.body.angular_velocity if hasattr(hit.shape.body, 'angular_velocity') else 0.0
                ray_scans.append(distance)
                ray_scans.append(arc_speed)
            else:
                ray_scans.append(1.0)
                ray_scans.append(0.0)
        return np.array(stuffnthings + ray_scans, dtype=np.float32)
# Collision handlers
    def mark_visited(self, objects, body: pymunk.Body):
        for object in objects:
            if (object[0] == body):
                object[2] = True
                if body not in self.object_visit_order:
                    self.object_visit_order.append(body)
                break

    def freeze_ball(self, space, key, data):
                self.ball_body.body_type = pymunk.Body.STATIC
                self.ball_body.velocity = (0, 0)
                self.ball_body.angular_velocity = 0
                self.ball_body.force = (0, 0)
                self.ball_body.torque = 0
    def unfreeze_ball(self, space, key, data):
            self.ball_body.body_type = pymunk.Body.DYNAMIC
            self.ball_body.mass = self.ball_body.data[0]
            self.ball_body.moment = self.ball_body.data[1]
    def set_jump_true(self, arbiter, space, data):
        self.can_jump = True
    def set_jump_false(self, arbiter, space, data):
        self.can_jump = False

    def reset_world(self, space: pymunk.Space):
        self.object_visit_order = []
        self.num_frames_passed = 0
        self.can_jump = False
        self.start[0].position = self.start[0].data[0]
        self.start[0].angle = self.start[0].data[1]
        self.start[0].angular_velocity = 0
        self.start[0].velocity = (0, 0)
        self.start[1].position = self.start[1].data[0]
        self.start[1].angle = self.start[1].data[1]
        self.start[1].angular_velocity = 0
        self.start[1].velocity = (0, 0)
        self.ball_body.position = (self.start[0].position.x + 7, self.start[0].position.y + 13)
        self.ball_body.velocity = (0, 0)
        self.ball_body.angle = 0
        self.ball_body.angular_velocity = 0
        self.ball_body.force = (0, 0)
        self.ball_body.torque = 0
        self.unfreeze_ball(space, key=self.ball_body, data={})
        for conveyor in self.conveyors:
            conveyor[0].position = conveyor[0].origin
            conveyor[0].velocity = (0, 0)
            conveyor[2] = False
            for spike in conveyor[0].spikes:
                spike[0].position = spike[0].origin[0]
                spike[0].angle = spike[0].origin[1]
                spike[0].velocity = (0, 0)
        for platform in self.platforms:
            platform[2] = False
        for arc in self.arcs:
            arc[0].angle = 0
            arc[2] = False
            for spike in arc[0].spikes:
                spike[0].position = spike[0].origin[0]
                spike[0].angle = spike[0].origin[1]
                spike[0].velocity = (0, 0)
        for hinge in self.hinges:
            hinge[0].angle = hinge[0].origin
            hinge[0].angular_velocity = 0
            hinge[2] = False
            for spike in hinge[0].spikes:
                spike[0].position = spike[0].origin[0]
                spike[0].angle = spike[0].origin[1]
                spike[0].angular_velocity = 0

    def end_fail(self, arbiter, space, data):
        self.win_state = -1
    def end_win(self, arbiter, space, data):
        self.win_state = 1

    def begin_platform(self, arbiter, space, data):
        _, platform_shape = arbiter.shapes
        self.mark_visited(self.platforms, platform_shape.body)

    def move_conveyor(self, arbiter, space, data):
        ball_shape, conveyor_shape = arbiter.shapes

        self.mark_visited(self.conveyors, conveyor_shape.body)
        if conveyor_shape.body.direction > 0:
            ball_shape.body.velocity = (conveyor_shape.body.move_speed, 0)
            conveyor_shape.body.velocity = (conveyor_shape.body.move_speed, 0)
            for spike in conveyor_shape.body.spikes:
                spike[0].velocity = (conveyor_shape.body.move_speed, 0)
        else:
            ball_shape.body.velocity = (-conveyor_shape.body.move_speed, 0)
            conveyor_shape.body.velocity = (-conveyor_shape.body.move_speed, 0)
            for spike in conveyor_shape.body.spikes:
                spike[0].velocity = (-conveyor_shape.body.move_speed, 0)
    def check_conveyor_state(self, arbiter, space, data):
        self.set_jump_true(arbiter, space, data)
        _, conveyor_shape = arbiter.shapes
        if conveyor_shape.body.position.x * conveyor_shape.body.direction > conveyor_shape.body.target * conveyor_shape.body.direction:
            conveyor_shape.body.velocity = (0, 0)
            for spike in conveyor_shape.body.spikes:
                spike[0].velocity = (0, 0)
    def stop_movement(self, arbiter, space, data):
        self.set_jump_false(arbiter, space, data)
        _, conveyor_shape = arbiter.shapes
        conveyor_shape.body.velocity = (0, 0)
        for spike in conveyor_shape.body.spikes:
            spike[0].velocity = (0, 0)

    def check_arc_state(self, arbiter, space, data):
        ball_body, arc_body = arbiter.bodies
        self.mark_visited(self.arcs, arc_body)
        if abs(ball_body.position.x - arc_body.position.x) < 0.35 * arc_body.radius and ((ball_body.position.y > arc_body.position.y) != (ball_body.position.get_distance(arc_body.position) < arc_body.radius)):
            self.set_jump_true(arbiter, space, data)
            distance = ball_body.position.get_distance(arc_body.position)
            direction = (ball_body.position - arc_body.position).normalized()
            if (ball_body.position.y > arc_body.position.y):
                correct_distance = (arc_body.radius + arc_body.thickness/2) + ball_body.radius
            else:
                correct_distance = (arc_body.radius - arc_body.thickness/2) - ball_body.radius
            ball_body.position += (correct_distance - distance) * direction
            space.add_post_step_callback(self.freeze_ball, key=ball_body, data={})
    def separate_arc(self, arbiter, space, data):
        ball_body, arc_body = arbiter.bodies
        if abs(ball_body.position.x - arc_body.position.x) < 0.4 * arc_body.radius:
            self.set_jump_false(arbiter, space, data)
            
            space.add_post_step_callback(self.unfreeze_ball, key=ball_body, data={})

    def move_hinge(self, arbiter, space, data):
        self.set_jump_true(arbiter, space, data)
        _, hinge_shape = arbiter.shapes
        self.mark_visited(self.hinges, hinge_shape.body)
        if not fcs.find_lowest_angle(hinge_shape.body.angle, hinge_shape.body.target):
            hinge_shape.body.angular_velocity = hinge_shape.body.speed * hinge_shape.body.direction
            for spike in hinge_shape.body.spikes:
                spike[0].angular_velocity = hinge_shape.body.speed * hinge_shape.body.direction
# End of collision handlers
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.win_state = 0
        self.reset_world(self.space)

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()
        return obs, {}

    def render(self):
        if self.render_mode == 'human':
            # Event handlers
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            self.screen.blit(self.canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(60)
        elif self.render_mode == 'rgb_array':
            return self._get_obs()
        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")
    def _draw(self):
        # Rendering
        for spike in self.spikes:
            pygame.draw.polygon(self.canvas, pygame.Color('black'), [(v.x, fcs.flipy(v.y)) for v in [spike[0].local_to_world(v) for v in spike[1].get_vertices()]])
    
        pygame.draw.circle(self.canvas, pygame.Color('white'), 
                            (int(self.ball_shape.body.position.x), 
                            int(fcs.flipy(self.ball_shape.body.position.y))), 
                            int(self.ball_shape.radius))
        pygame.draw.circle(self.canvas, pygame.Color('black'), 
                            (int(self.ball_shape.body.position.x), 
                            int(fcs.flipy(self.ball_shape.body.position.y))), 
                            int(self.ball_shape.radius), 2)
        
        for platform in self.platforms:
            pygame.draw.polygon(self.canvas, pygame.Color('black'), [(v.x, fcs.flipy(v.y)) for v in [platform[0].local_to_world(v) for v in platform[1].get_vertices()]])
    
        for conveyor in self.conveyors:
            pygame.draw.polygon(self.canvas, pygame.Color('white'), [(v.x, fcs.flipy(v.y)) for v in [conveyor[0].local_to_world(v) for v in conveyor[1].get_vertices()]])
        
        for hinge in self.hinges:
            pygame.draw.polygon(self.canvas, (111, 237, 220), [(v.x, fcs.flipy(v.y)) for v in [hinge[0].local_to_world(v) for v in hinge[1].get_vertices()]])
            pygame.draw.circle(self.canvas, (111, 237, 220), (int(hinge[0].pivot_position.x), int(fcs.flipy(hinge[0].pivot_position.y))), hinge[0].width)
    
        for arc in self.arcs:
            for segment in arc[1]:
                pygame.draw.polygon(self.canvas, (27, 245, 39), [(v.x, fcs.flipy(v.y)) for v in [arc[0].local_to_world(v) for v in segment.get_vertices()]])
    
        pygame.draw.polygon(self.canvas, pygame.Color('black'), [(v.x, fcs.flipy(v.y)) for v in [self.start[0].local_to_world(v) for v in self.start[2].get_vertices()]])
        pygame.draw.polygon(self.canvas, pygame.Color('black'), [(v.x, fcs.flipy(v.y)) for v in [self.start[1].local_to_world(v) for v in self.start[3].get_vertices()]])
        pygame.draw.circle(self.canvas, pygame.Color('black'), (int(self.start[0].pivot_position.x), int(fcs.flipy(self.start[0].pivot_position.y))), self.start[0].width/2)
        pygame.draw.circle(self.canvas, pygame.Color('black'), (int(self.start[1].pivot_position.x), int(fcs.flipy(self.start[1].pivot_position.y))), self.start[1].width/2)
    
        for segment in self.goal[0][1]:
            a = self.goal[0][0].local_to_world(segment.a)
            b = self.goal[0][0].local_to_world(segment.b)
            pygame.draw.line(self.canvas, pygame.Color('black'), (a.x, fcs.flipy(a.y)), (b.x, fcs.flipy(b.y)), width=int(segment.radius*2))
    def close(self):
        pygame.quit()

    def step(self, action):
        self.num_frames_passed += 1
        reward = 0.0
        if action == 1:
            if self.can_jump:
                self.unfreeze_ball(self.space, key=self.ball_body, data={})
                self.ball_body.velocity = (self.ball_body.velocity.x, 0.4 * self.GRAVITY)
            else:
                reward -= 2.0
    
        for hinge in self.hinges:
            if fcs.find_lowest_angle(hinge[0].angle, hinge[0].target):
                hinge[0].angular_velocity = 0
                for spike in hinge[0].spikes:
                    spike[0].angular_velocity = 0
        self.canvas.fill((68, 156, 144))

        
        # Advance physics engine
        dt = 1.0 / 60.0
        for x in range(1):
            self.space.step(dt)
        
        # Reset when ball leaves screen
        # Theoretically should never run due to the border, but just in case something weird happens, still might be useful.

        if self.render_mode is not None:
            self.canvas.fill((68, 156, 144))
            self._draw()
            if self.render_mode == 'human':
                self.render()
        
        observation = self._get_obs()
        info = {}
        terminated = self.win_state != 0
        truncated = self.num_frames_passed > 3000

        if self.win_state != 0:
           reward = self.win_state * 100

        return observation, reward, terminated, truncated, info

gym.register(id="gymnasium_env/GoEscape-v0",
             entry_point=GoEscapeEnv,
             max_episode_steps=30000)
# hi!
if __name__ == "__main__":

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = make_vec_env(GoEscapeEnv, n_envs=8, env_kwargs={"render_mode": None, "ray_num": 24})
    
    model = PPO("MlpPolicy", env, verbose=1, n_steps=512, device="cuda")
    print("starting")
    model.learn(total_timesteps=300000, progress_bar=True)
    model.save("ppo_goEscape")
    print("trained!")
