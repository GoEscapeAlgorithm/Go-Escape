import pymunk
SCREEN_DIMENSIONS = (300, 600)
# Collision types:
# BALL_TYPE = 0
# STATIC_TERRAIN_TYPE = 1
# SPIKE_TYPE = 2
# GOAL_TYPE = 3
# CONVEYOR_TYPE = 4

def flipy(y):
    return -y + SCREEN_DIMENSIONS[1]

def get_world_coords(shape: pymunk.Shape):
    coords = []
    for v in shape.get_vertices():
        x, y = v.rotated(shape.body.angle) + shape.body.position
        coords.append([x, y])
    return coords

def add_platform(space: pymunk.Space, x: int, y: int, angle: float, width = 120, height = 10):
    platform_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    platform_body.position = (x, y)
    platform_body.angle = angle
    platform_shape = pymunk.Poly.create_box(platform_body, (width, height))
    platform_shape.collision_type = 1 # Static terrain collision
    space.add(platform_body, platform_shape)
    return platform_body, platform_shape

def add_platform_spike(space: pymunk.Space, platform_shape: pymunk.Shape, size = 10, num_spikes = 1, higher = False):
    platform_coords = get_world_coords(platform_shape)
    if higher:
        platform_coords = max(platform_coords, key=lambda point: point[1])
    else:
        if platform_shape.body.angle >= 0:
            platform_coords = min(platform_coords, key=lambda point: point[0])
        else:
            platform_coords = max(platform_coords, key=lambda point: point[0])
    spikes = []

    base_coord_1 = pymunk.Vec2d.from_polar(size, -2.6180)
    base_coord_2 = pymunk.Vec2d.from_polar(size, -0.5236)
    peak_coord = pymunk.Vec2d.from_polar(size, 1.5708)

    for i in range(num_spikes):
        spike_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        spike_body.position = pymunk.Vec2d(platform_coords[0], platform_coords[1]) + pymunk.Vec2d.from_polar(size, (platform_shape.body.angle + 0.5236 if platform_shape.body.angle >= 0 ^ higher else platform_shape.body.angle + 2.6180))

        spike_body.angle = platform_shape.body.angle
        spike_shape = pymunk.Poly(spike_body, [base_coord_1, base_coord_2, peak_coord])
        spike_shape.collision_type = 2 # spike collision (currently static terrain for testing)
        space.add(spike_body, spike_shape)
        spikes.append([spike_body, spike_shape])

        platform_coords += pymunk.Vec2d.from_polar(1.5 * size, platform_shape.body.angle if not higher else platform_shape.body.angle + 3.14159)
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

def add_conveyor(space: pymunk.Space, x: int, y: int, velocity: float, target: int, width = 120, height = 10):
    conveyor_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    conveyor_body.position = (x, y)
    conveyor_body.origin = (x, y)
    conveyor_body.velocity = (0, 0)
    conveyor_body.move_speed = velocity
    conveyor_body.target = target
    conveyor_body.spikes = []
    conveyor_body.direction = -1 if x > target else 1
    conveyor_shape = pymunk.Poly.create_box(conveyor_body, (width, height))
    conveyor_shape.collision_type = 4 # conveyor collision
    space.add(conveyor_body, conveyor_shape)
    return conveyor_body, conveyor_shape

def add_conveyor_spike(space: pymunk.Space, conveyor_shape: pymunk.Shape, size = 10, num_spikes = 1, left=True):
    conveyor_coords = get_world_coords(conveyor_shape)
    if left:
        conveyor_coords = min(conveyor_coords, key=lambda point: (point[0], -point[1]))
    else:
        conveyor_coords = max(conveyor_coords, key=lambda point: (point[0], point[1]))

    base_coord_1 = pymunk.Vec2d.from_polar(size, -2.6180)
    base_coord_2 = pymunk.Vec2d.from_polar(size, -0.5236)
    peak_coord = pymunk.Vec2d.from_polar(size, 1.5708)

    for i in range(num_spikes):
        spike_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        spike_body.position = pymunk.Vec2d(conveyor_coords[0], conveyor_coords[1]) + pymunk.Vec2d.from_polar(size, (conveyor_shape.body.angle + 0.5236 if left else conveyor_shape.body.angle + 2.6180))
        spike_body.angle = conveyor_shape.body.angle
        spike_body.origin = spike_body.position
        spike_shape = pymunk.Poly(spike_body, [base_coord_1, base_coord_2, peak_coord])
        spike_shape.collision_type = 2 # spike collision (currently static terrain for testing)
        space.add(spike_body, spike_shape)
        conveyor_shape.body.spikes.append([spike_body, spike_shape])

        conveyor_coords += pymunk.Vec2d.from_polar(1.5 * size, 0 if left else 3.14159)
    if num_spikes == 1:
        return conveyor_shape.body.spikes[0]
    else:
        return conveyor_shape.body.spikes