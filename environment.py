import pygame
import pymunk

SCREEN_DIMENSIONS = (300, 600)
GRAVITY = 450

def flipy(y):
    return -y + SCREEN_DIMENSIONS[1]

def get_world_coords(shape: pymunk.Shape):
    coords = []
    for v in shape.get_vertices():
        x, y = v.rotated(shape.body.angle) + shape.body.position
        coords.append([x, y])
    return coords

def add_platform(space: pymunk.Space, x: int, y: int, angle: float, width: int, height: int):
    platform_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    platform_body.position = (x, y)
    platform_body.angle = angle
    platform_shape = pymunk.Poly.create_box(platform_body, (width, height))
    platform_shape.collision_type = STATIC_TERRAIN_TYPE
    space.add(platform_body, platform_shape)
    return platform_body, platform_shape

def add_platform_spike(space: pymunk.Space, platform_shape: pymunk.Shape, size: int, num_spikes = 1, higher = False):
    platform_coords = get_world_coords(platform_shape)
    if higher:
        platform_coords = max(platform_coords, key=lambda point: point[1])
    else:
        if platform_shape.body.angle >= 0:
            platform_coords = min(platform_coords, key=lambda point: point[0])
        else:
            platform_coords = max(platform_coords, key=lambda point: point[0])
    spikes = []

    main_coord = pymunk.Vec2d.from_polar(size, -2.6180)
    other_base_coord = pymunk.Vec2d.from_polar(size, -0.5236)
    peak_coord = pymunk.Vec2d.from_polar(size, 1.5708)

    for i in range(num_spikes):
        spike_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        spike_body.position = pymunk.Vec2d(platform_coords[0], platform_coords[1]) + pymunk.Vec2d.from_polar(size, (platform_shape.body.angle + 0.5236 if platform_shape.body.angle >= 0 else platform_shape.body.angle + 2.6180))

        spike_body.angle = platform_shape.body.angle
        spike_shape = pymunk.Poly(spike_body, [main_coord, other_base_coord, peak_coord])
        spike_shape.collision_type = STATIC_TERRAIN_TYPE
        space.add(spike_body, spike_shape)
        spikes.append([spike_body, spike_shape])

        platform_coords += pymunk.Vec2d.from_polar(1.5 * size, platform_shape.body.angle)
    if num_spikes == 1:
        return spikes[0]
    else:
        return spikes

def get_closest_spike(space: pymunk.Space, ball_body: pymunk.Body, spikes):
    spike_coords = []
    for spike in spikes:
        spike_coords.append(spike[0].position)
    ball_coords = ball_body.position

    closest = spike_coords[0]
    for coords in spike_coords:
        if ball_coords.get_distance(coords) < ball_coords.get_distance(closest):
            closest = coords
    return closest, ball_coords.get_distance(closest)



pygame.init()
screen = pygame.display.set_mode(SCREEN_DIMENSIONS)
clock = pygame.time.Clock()
running = True
platforms = []
spikes = []

space = pymunk.Space()
space.gravity = 0.0, -GRAVITY
BALL_TYPE = 0
STATIC_TERRAIN_TYPE = 1
SPIKE_TYPE = 2
GOAL_TYPE = 3

ball_body = pymunk.Body(mass = 10, moment = 1, body_type=pymunk.Body.DYNAMIC)
ball_body.position = (150, 600)
ball_shape = pymunk.Circle(ball_body, 10)
ball_shape.collision_type = BALL_TYPE
space.add(ball_body, ball_shape)

platforms.append(add_platform(space, 150, 500, 0.25, 120, 10))
platforms.append(add_platform(space, 0, 300, -0.5, 120, 10))
spikes.extend(add_platform_spike(space, platforms[0][1], 10, 2))
spikes.append(add_platform_spike(space, platforms[1][1], 10))

def end_sim(arbiter, space, data):
    global running
    running = False

can_jump = False
def flip_jump_state(arbiter, space, data):
    global can_jump
    can_jump = not can_jump
space.on_collision(BALL_TYPE, STATIC_TERRAIN_TYPE, begin=flip_jump_state, separate=flip_jump_state)
space.on_collision(BALL_TYPE, SPIKE_TYPE, begin=end_sim)


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
                       int(flipy(ball_shape.body.position.y))), 
                       int(ball_shape.radius), 2)
    
    for platform in platforms:
        pygame.draw.polygon(screen, pygame.Color('black'), [(v.x, flipy(v.y)) for v in [platform[0].local_to_world(v) for v in platform[1].get_vertices()]])

    for spike in spikes:
        pygame.draw.polygon(screen, pygame.Color('black'), [(v.x, flipy(v.y)) for v in [spike[0].local_to_world(v) for v in spike[1].get_vertices()]])
        screen.set_at((int(spike[0].position.x), int(spike[0].position.y)), pygame.Color('red')) 

    dt = 1.0 / 60.0
    for x in range(1):
        space.step(dt)
    
    if ball_body.position[1] < -50:
        ball_body.position = (150, 600)
        ball_body.velocity = (0, 0)
    pygame.display.flip()

    clock.tick(60)

pygame.quit()
print(get_closest_spike(space, ball_body, spikes))