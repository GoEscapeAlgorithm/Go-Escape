import pymunk
import pymunk.autogeometry
SCREEN_DIMENSIONS = (300, 600)
# Collision types:
# BALL_TYPE = 0
# STATIC_TERRAIN_TYPE = 1
# SPIKE_TYPE = 2
# GOAL_TYPE = 3
# CONVEYOR_TYPE = 4
# ARC_TYPE = 5

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
    platform_body.spikes = []
    platform_shape = pymunk.Poly.create_box(platform_body, (width, height))
    platform_shape.collision_type = 1 # Static terrain collision
    space.add(platform_body, platform_shape)
    return [[platform_body, platform_shape, False]]

def add_platform_spike(space: pymunk.Space, platform_shape: pymunk.Shape, size = 10, num_spikes = 1, higher = False, offset = 0):
    platform_coords = get_world_coords(platform_shape)
    if higher:
        platform_coords = max(platform_coords, key=lambda point: point[1])
    else:
        if platform_shape.body.angle >= 0:
            platform_coords = min(platform_coords, key=lambda point: point[0])
        else:
            platform_coords = max(platform_coords, key=lambda point: point[0])

    platform_coords += pymunk.Vec2d.from_polar(offset, platform_shape.body.angle if not higher else platform_shape.body.angle + 3.14159)
    base_coord_1 = pymunk.Vec2d.from_polar(size, -2.6180)
    base_coord_2 = pymunk.Vec2d.from_polar(size, -0.5236)
    peak_coord = pymunk.Vec2d.from_polar(size, 1.5708)

    for i in range(num_spikes):
        spike_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        spike_body.position = pymunk.Vec2d(platform_coords[0], platform_coords[1]) + pymunk.Vec2d.from_polar(size, (platform_shape.body.angle + 0.5236 if platform_shape.body.angle >= 0 ^ higher else platform_shape.body.angle + 2.6180))

        spike_body.angle = platform_shape.body.angle
        spike_shape = pymunk.Poly(spike_body, [base_coord_1, base_coord_2, peak_coord])
        spike_shape.collision_type = 2 # Spike collision 
        space.add(spike_body, spike_shape)
        platform_shape.body.spikes.append([spike_body, spike_shape])

        platform_coords += pymunk.Vec2d.from_polar(1.5 * size, platform_shape.body.angle if not higher else platform_shape.body.angle + 3.14159)

    return platform_shape.body.spikes

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
    conveyor_shape.collision_type = 4 # Conveyor collision
    space.add(conveyor_body, conveyor_shape)
    return [[conveyor_body, conveyor_shape, False]]

def add_conveyor_spike(space: pymunk.Space, conveyor_shape: pymunk.Shape, size = 10, num_spikes = 1, left = True, offset = 0):
    conveyor_coords = get_world_coords(conveyor_shape)
    if left:
        conveyor_coords = min(conveyor_coords, key=lambda point: (point[0], -point[1]))
    else:
        conveyor_coords = max(conveyor_coords, key=lambda point: (point[0], point[1]))
    conveyor_coords += pymunk.Vec2d.from_polar(offset, 0 if left else 3.14159)

    base_coord_1 = pymunk.Vec2d.from_polar(size, -2.6180)
    base_coord_2 = pymunk.Vec2d.from_polar(size, -0.5236)
    peak_coord = pymunk.Vec2d.from_polar(size, 1.5708)

    for i in range(num_spikes):
        spike_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        spike_body.position = pymunk.Vec2d(conveyor_coords[0], conveyor_coords[1]) + pymunk.Vec2d.from_polar(size, (conveyor_shape.body.angle + 0.5236 if left else conveyor_shape.body.angle + 2.6180))
        spike_body.angle = conveyor_shape.body.angle
        spike_body.origin = [spike_body.position, spike_body.angle]
        spike_shape = pymunk.Poly(spike_body, [base_coord_1, base_coord_2, peak_coord])
        spike_shape.collision_type = 2 # Spike collision 
        space.add(spike_body, spike_shape)
        conveyor_shape.body.spikes.append([spike_body, spike_shape])

        conveyor_coords += pymunk.Vec2d.from_polar(1.5 * size, 0 if left else 3.14159)
    return conveyor_shape.body.spikes

def add_goal(space: pymunk.Space, x: int, y: int, width = 100, line_thickness = 10):
    goal_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    goal_body.position = (x, y)
    goal_base_shape = pymunk.Segment(goal_body, 
                                     pymunk.Vec2d.from_polar(width/2, -2.0944), 
                                     pymunk.Vec2d.from_polar(width/2, -1.0472),
                                     radius=line_thickness/2)
    goal_left_wall_shape = pymunk.Segment(goal_body, 
                                          pymunk.Vec2d.from_polar(width/2, 3.1416),
                                          pymunk.Vec2d.from_polar(width/2, -2.0944),
                                          radius=line_thickness/2)
    goal_right_wall_shape = pymunk.Segment(goal_body,
                                           pymunk.Vec2d.from_polar(width/2, -1.0472),
                                           pymunk.Vec2d.from_polar(width/2, 0),
                                           radius=line_thickness/2)
    goal_base_shape.collision_type = 3 # Goal collision
    goal_left_wall_shape.collision_type = 1 # Static terrain collision
    goal_right_wall_shape.collision_type = 1 # Static terrain collision
    space.add(goal_body, goal_base_shape, goal_left_wall_shape, goal_right_wall_shape)
    return [[goal_body, [goal_base_shape, goal_left_wall_shape, goal_right_wall_shape]]]

def add_arc(space: pymunk.Space, x: int, y: int, radius: int, start_angle: float, end_angle: float, speed = 1.25, segments=100, thickness=8):
    start_angle = 6.2832 + start_angle if start_angle < 0 else start_angle
    end_angle = 6.2832 + end_angle if end_angle < 0 else end_angle

    arc_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    arc_body.position = (x, y)
    arc_body.angular_velocity = speed
    arc_body.radius = radius
    arc_body.start_angle = start_angle
    arc_body.end_angle = end_angle
    arc_body.spikes = []
    space.add(arc_body)

    angle_step = abs(end_angle - start_angle) / segments

    shapes = []
    for rah in range(segments):
        angle1 = start_angle + (rah * angle_step)
        angle2 = start_angle + (rah + 1) * angle_step
        vertices = []
        vertices.append(pymunk.Vec2d.from_polar(radius-thickness/2, angle1))
        vertices.append(pymunk.Vec2d.from_polar(radius+thickness/2, angle1))
        vertices.append(pymunk.Vec2d.from_polar(radius-thickness/2, angle2))
        vertices.append(pymunk.Vec2d.from_polar(radius+thickness/2, angle2))
        piece_shape = pymunk.Poly(arc_body, vertices)
        piece_shape.friction = 0
        piece_shape.collision_type = 5 # Arc collision
        space.add(piece_shape)
        shapes.append(piece_shape)
    
    return [[arc_body, shapes, False]]

def add_arc_spike(space: pymunk.Space, arc_body: pymunk.Body, size = 17, num_spikes = 1, offset = 0, from_start = True, outside = True):
    offset = offset / arc_body.radius
    offset = (offset + 0.79 * size / arc_body.radius if from_start else -offset - 0.79 * size / arc_body.radius)
    for i in range(num_spikes):
        spike_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        spike_body.position = arc_body.position
        spike_body.angle = (arc_body.start_angle if from_start else arc_body.end_angle)
        spike_body.origin = [spike_body.position, spike_body.angle]
        spike_body.angular_velocity = arc_body.angular_velocity
        spike_body.geo_center = pymunk.Vec2d.from_polar(arc_body.radius, offset)
        base_coord_1 = pymunk.Vec2d.from_polar(size/2, offset + 1.5708) + spike_body.geo_center
        base_coord_2 = pymunk.Vec2d.from_polar(size/2, offset - 1.5708) + spike_body.geo_center
        peak_coord = pymunk.Vec2d.from_polar(size, offset + (0 if outside else 3.14159)) + spike_body.geo_center
        spike_shape = pymunk.Poly(spike_body, [base_coord_1, base_coord_2, peak_coord])
        spike_shape.collision_type = 2 # Spike collision
        space.add(spike_body, spike_shape)
        arc_body.spikes.append([spike_body, spike_shape])

        a = 0.6 if outside else 0.8
        offset = offset + (a * size / arc_body.radius if from_start else a * -size / arc_body.radius)
    return arc_body.spikes
