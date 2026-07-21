import pygame
import pymunk
import env_functions as fcs

SCREEN_DIMENSIONS = (300, 600)
GRAVITY = 450
BALL_TYPE = 0
STATIC_TERRAIN_TYPE = 1
SPIKE_TYPE = 2
GOAL_TYPE = 3
CONVEYOR_TYPE = 4


pygame.init()
screen = pygame.display.set_mode(SCREEN_DIMENSIONS)
clock = pygame.time.Clock()
running = True
can_jump = False
all_objects = []
platforms = []
spikes = []
conveyors = []

space = pymunk.Space()
space.gravity = 0.0, -GRAVITY

ball_body = pymunk.Body(mass = 10, moment = 1, body_type=pymunk.Body.DYNAMIC)
ball_body.position = (150, 600)
ball_shape = pymunk.Circle(ball_body, 10)
ball_shape.collision_type = BALL_TYPE
space.add(ball_body, ball_shape)

platforms.append(fcs.add_platform(space, 150, 500, 0.25))
platforms.append(fcs.add_platform(space, 0, 300, -0.5))
spikes.extend(fcs.add_platform_spike(space, platforms[0][1], num_spikes=2))
spikes.append(fcs.add_platform_spike(space, platforms[1][1]))
conveyors.append(fcs.add_conveyor(space, 200, 150, 100, 100))
spikes.extend(fcs.add_conveyor_spike(space, conveyors[0][1], num_spikes=2))

all_objects.extend(platforms)
all_objects.extend(spikes)
all_objects.extend(conveyors)

def reset_world(space: pymunk.Space):
    ball_body.position = (150, 600)
    ball_body.velocity = (0, 0)
    for conveyor in conveyors:
        conveyor[0].position = conveyor[0].origin
        conveyor[0].velocity = (0, 0)
        for spike in conveyor[0].spikes:
            spike[0].position = spike[0].origin
            spike[0].velocity = (0, 0)

def end_fail(arbiter, space, data):
    reset_world(space)
def end_win(arbiter, space, data):
    reset_world(space)

def flip_jump_state(arbiter, space, data):
    global can_jump
    can_jump = not can_jump

def move_conveyor(arbiter, space, data):
    global can_jump
    can_jump = not can_jump
    ball_shape, conveyor_shape = arbiter.shapes
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
        conveyor_shape.body.velocity = (0, 0)
        for spike in conveyor_shape.body.spikes:
            spike[0].velocity = (0, 0)

space.on_collision(BALL_TYPE, STATIC_TERRAIN_TYPE, begin=flip_jump_state, separate=flip_jump_state)
space.on_collision(BALL_TYPE, SPIKE_TYPE, begin=end_fail)
space.on_collision(BALL_TYPE, CONVEYOR_TYPE, begin=move_conveyor, pre_solve=check_conveyor_state, separate=flip_jump_state)


while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif can_jump and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            ball_body.velocity = (ball_body.velocity.x, 0.4 * GRAVITY)
    
    screen.fill('white')

    # render
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

    dt = 1.0 / 60.0
    for x in range(1):
        space.step(dt)
    if ball_body.position[1] < -50:
        reset_world(space)
    pygame.display.flip()

    clock.tick(60)

pygame.quit()