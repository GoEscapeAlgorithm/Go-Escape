import pygame
import pymunk
import env_functions as fcs

# General settings
SCREEN_DIMENSIONS = (300, 600)
GRAVITY = 450
BALL_TYPE = 0
STATIC_TERRAIN_TYPE = 1
SPIKE_TYPE = 2
GOAL_TYPE = 3
CONVEYOR_TYPE = 4
ARC_TYPE = 5


# Pygame and pymunk initialization
pygame.init()
screen = pygame.display.set_mode(SCREEN_DIMENSIONS)
clock = pygame.time.Clock()
running = True
space = pymunk.Space()
space.gravity = 0.0, -GRAVITY
space.iterations = 20
space.idle_speed_threshold = 0.2

# Level and state storage
can_jump = False
all_objects = []
platforms = []
spikes = []
conveyors = []
goal = []
arcs = []

# Ball initialization
ball_body = pymunk.Body(mass = 10, moment = 1, body_type=pymunk.Body.DYNAMIC)
ball_body.position = (150, 600)
ball_body.radius = 10
ball_body.data = ball_body.mass, ball_body.moment
ball_shape = pymunk.Circle(ball_body, 10)
ball_shape.collision_type = BALL_TYPE
space.add(ball_body, ball_shape)

# World border initialization
border_body = space.static_body
bottom_shape = pymunk.Segment(border_body, (-SCREEN_DIMENSIONS[0], -60), (SCREEN_DIMENSIONS[0] * 2, -SCREEN_DIMENSIONS[0]), radius=10)
bottom_shape.collision_type = SPIKE_TYPE
space.add(bottom_shape)

# Level Creation - use extend instead of append
platforms.extend(fcs.add_platform(space, 150, 500, 0.25))
platforms.extend(fcs.add_platform(space, 0, 300, -0.5))
spikes.extend(fcs.add_platform_spike(space, platforms[0][1], num_spikes=2))
spikes.extend(fcs.add_platform_spike(space, platforms[1][1]))
conveyors.extend(fcs.add_conveyor(space, 200, 150, 100, 150))
spikes.extend(fcs.add_conveyor_spike(space, conveyors[0][1], left=False, num_spikes=2))
arcs.extend(fcs.add_arc(space, 150, 250, 50, +0.5236, -0.5236))
spikes.extend(fcs.add_arc_spike(space, arcs[0][0], num_spikes=2))
spikes.extend(fcs.add_arc_spike(space, arcs[0][0], num_spikes=2, from_start=False, outside=False, offset=75))
goal.extend(fcs.add_goal(space, 50, 50))

all_objects.extend(platforms)
all_objects.extend(spikes)
all_objects.extend(conveyors)
all_objects.extend(arcs)
all_objects.extend(goal)

# Collision handlers
def mark_visited(objects, shape: pymunk.Shape):
    for object in objects:
        if (object[1] == shape):
            object[2] = True
            break

def freeze_ball(space, key, data):
            ball_body.body_type = pymunk.Body.STATIC
            ball_body.velocity = (0, 0)
            ball_body.angular_velocity = 0
            ball_body.force = (0, 0)
            ball_body.torque = 0
def unfreeze_ball(space, key, data):
        ball_body.body_type = pymunk.Body.DYNAMIC
        ball_body.mass = ball_body.data[0]
        ball_body.moment = ball_body.data[1]

def reset_world(space: pymunk.Space, key, data):
    space.remove(ball_body, ball_shape)

    ball_body.position = (150, 600)
    ball_body.velocity = (0, 0)
    unfreeze_ball(space, key=ball_body, data={})

    space.add(ball_body, ball_shape)
    global can_jump
    can_jump = False
    for conveyor in conveyors:
        conveyor[0].position = conveyor[0].origin
        conveyor[0].velocity = (0, 0)
        conveyor[2] = False
        for spike in conveyor[0].spikes:
            spike[0].position = spike[0].origin[0]
            spike[0].angle = spike[0].origin[1]
            spike[0].velocity = (0, 0)
    for platform in platforms:
        platform[2] = False
    for arc in arcs:
        arc[0].angle = 0
        arc[2] = False
        for spike in arc[0].spikes:
            spike[0].position = spike[0].origin[0]
            spike[0].angle = spike[0].origin[1]
            spike[0].velocity = (0, 0)

def end_fail(arbiter, space, data):
    space.add_post_step_callback(reset_world, key="reset", data={})
def end_win(arbiter, space, data):
    space.add_post_step_callback(reset_world, key="reset", data={})

def begin_platform(arbiter, space, data):
    global can_jump
    can_jump = True
    _, platform_shape = arbiter.shapes
    mark_visited(platforms, platform_shape)
def separate_platform(arbiter, space, data):
    global can_jump
    can_jump = False

def move_conveyor(arbiter, space, data):
    global can_jump
    can_jump = True
    ball_shape, conveyor_shape = arbiter.shapes

    mark_visited(conveyors, conveyor_shape)
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
def check_conveyor_state(arbiter, space, data):
    _, conveyor_shape = arbiter.shapes
    if conveyor_shape.body.position.x * conveyor_shape.body.direction > conveyor_shape.body.target * conveyor_shape.body.direction:
        stop_movement(arbiter, space, data)
        global can_jump
        can_jump = True
def stop_movement(arbiter, space, data):
    global can_jump
    can_jump = False 
    _, conveyor_shape = arbiter.shapes
    conveyor_shape.body.velocity = (0, 0)
    for spike in conveyor_shape.body.spikes:
        spike[0].velocity = (0, 0)

def check_arc_state(arbiter, space, data):
    ball_body, arc_body = arbiter.bodies
    mark_visited(arcs, arbiter.shapes[1])
    if abs(ball_body.position.x - arc_body.position.x) < 0.4 * arc_body.radius:
        global can_jump
        can_jump = True
        
        space.add_post_step_callback(freeze_ball, key=ball_body, data={})
def separate_arc(arbiter, space, data):
    global can_jump
    can_jump = False
    
    space.add_post_step_callback(unfreeze_ball, key=ball_body, data={})
space.on_collision(BALL_TYPE, STATIC_TERRAIN_TYPE, begin=begin_platform, separate=separate_platform)
space.on_collision(BALL_TYPE, CONVEYOR_TYPE, begin=move_conveyor, pre_solve=check_conveyor_state, separate=stop_movement)
space.on_collision(BALL_TYPE, SPIKE_TYPE, begin=end_fail)
space.on_collision(BALL_TYPE, GOAL_TYPE, begin=end_win)
space.on_collision(BALL_TYPE, ARC_TYPE, pre_solve=check_arc_state, separate=separate_arc)

# Main game loop
while running:
    # Event handlers
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
    keys = pygame.key.get_pressed()
    
    if can_jump and keys[pygame.K_SPACE]:
        unfreeze_ball(space, key=ball_body, data={})
        ball_body.velocity = (ball_body.velocity.x, 0.4 * GRAVITY)
    
    screen.fill('white')

    # Rendering
    pygame.draw.circle(screen, pygame.Color('black'), 
                       (int(ball_shape.body.position.x), 
                       int(fcs.flipy(ball_shape.body.position.y))), 
                       int(ball_shape.radius), 2)
    
    for platform in platforms:
        pygame.draw.polygon(screen, pygame.Color('black'), [(v.x, fcs.flipy(v.y)) for v in [platform[0].local_to_world(v) for v in platform[1].get_vertices()]])

    for conveyor in conveyors:
        pygame.draw.polygon(screen, pygame.Color('black'), [(v.x, fcs.flipy(v.y)) for v in [conveyor[0].local_to_world(v) for v in conveyor[1].get_vertices()]])
    
    for spike in spikes:
        pygame.draw.polygon(screen, pygame.Color('black'), [(v.x, fcs.flipy(v.y)) for v in [spike[0].local_to_world(v) for v in spike[1].get_vertices()]])

    for arc in arcs:
        for segment in arc[1]:
            pygame.draw.polygon(screen, pygame.Color('black'), [(v.x, fcs.flipy(v.y)) for v in [arc[0].local_to_world(v) for v in segment.get_vertices()]])

    for segment in goal[0][1]:
        start = goal[0][0].local_to_world(segment.a)
        end = goal[0][0].local_to_world(segment.b)
        pygame.draw.line(screen, pygame.Color('black'), (start.x, fcs.flipy(start.y)), (end.x, fcs.flipy(end.y)), width=int(segment.radius*2))
    
    # Advance physics engine
    dt = 1.0 / 60.0
    for x in range(1):
        space.step(dt)
    
    # Reset when ball leaves screen
    # Theoretically should never run due to the border, but just in case something weird happens, still might be useful. 
    if ball_body.position[1] < -50:
        reset_world(space)
    pygame.display.flip()

    clock.tick(60)

pygame.quit()