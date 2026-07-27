import pymunk
import random
SCREEN_DIMENSIONS = (300, 600)
# Collision types:
# BALL_TYPE = 0
# STATIC_TERRAIN_TYPE = 1
# SPIKE_TYPE = 2
# GOAL_TYPE = 3
# CONVEYOR_TYPE = 4
# ARC_TYPE = 5
# HINGE_TYPE = 6

def get_closest_object(ball_body: pymunk.Body, objects):
    closest = objects[0]
    ball_coords = ball_body.position
    for object in objects:
        if object[0].geo_center.get_distance(ball_coords) < closest[0].geo_center.get_distance(ball_coords):
            closest = object

    return closest, ball_coords.get_distance(closest[0].position)

def find_lowest_angle(angle1: float, angle2: float):
    vec_1 = pymunk.Vec2d.from_polar(1, angle1)
    vec_2 = pymunk.Vec2d.from_polar(1, angle2)
    vec_t = pymunk.Vec2d.from_polar(1, -1.5708)

    return vec_1.dot(vec_t) > vec_2.dot(vec_t)

def flipy(y):
    return -y + SCREEN_DIMENSIONS[1]

def get_world_coords(shape: pymunk.Shape):
    coords = []
    for v in shape.get_vertices():
        x, y = v.rotated(shape.body.angle) + shape.body.position
        coords.append([x, y])
    return coords

def add_platform(space: pymunk.Space, x: int, y: int, angle: float, width = 120, height = 5):
    platform_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    platform_body.position = (x, y)
    platform_body.geo_center = platform_body.position
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
        spike_body.geo_center = spike_body.position
        spike_body.angle = platform_shape.body.angle
        spike_shape = pymunk.Poly(spike_body, [base_coord_1, base_coord_2, peak_coord])
        spike_shape.collision_type = 2 # Spike collision 
        space.add(spike_body, spike_shape)
        platform_shape.body.spikes.append([spike_body, spike_shape])

        platform_coords += pymunk.Vec2d.from_polar(1.5 * size, platform_shape.body.angle if not higher else platform_shape.body.angle + 3.14159)

    return platform_shape.body.spikes

def add_conveyor(space: pymunk.Space, x: int, y: int, velocity: float, target: int, width = 120, height = 5):
    conveyor_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    conveyor_body.position = (x, y)
    conveyor_body.geo_center = conveyor_body.position
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
        spike_body.geo_center = spike_body.position
        spike_body.origin = [spike_body.position, spike_body.angle]
        spike_shape = pymunk.Poly(spike_body, [base_coord_1, base_coord_2, peak_coord])
        spike_shape.collision_type = 2 # Spike collision 
        space.add(spike_body, spike_shape)
        conveyor_shape.body.spikes.append([spike_body, spike_shape])

        conveyor_coords += pymunk.Vec2d.from_polar(1.5 * size, 0 if left else 3.14159)
    return conveyor_shape.body.spikes

def add_goal(space: pymunk.Space, x: int, y: int, width = 100, line_thickness = 5):
    goal_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    goal_body.position = (x, y)
    goal_body.geo_center = (x, y)
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
    space.add(goal_body, goal_base_shape, goal_left_wall_shape, goal_right_wall_shape)
    return [[goal_body, [goal_base_shape, goal_left_wall_shape, goal_right_wall_shape]]]

def add_arc(space: pymunk.Space, x: int, y: int, radius: int, start_angle = 0.5236, end_angle = -0.5236, speed = 1.5, segments=100, thickness=7):
    start_angle = 6.2832 + start_angle if start_angle < 0 else start_angle
    end_angle = 6.2832 + end_angle if end_angle < 0 else end_angle

    arc_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    arc_body.position = (x, y)
    arc_body.geo_center = arc_body.position
    arc_body.angular_velocity = speed
    arc_body.radius = radius
    arc_body.start_angle = start_angle
    arc_body.end_angle = end_angle
    arc_body.spikes = []
    arc_body.thickness = thickness
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

def add_arc_spike(space: pymunk.Space, arc_body: pymunk.Body, size = 13, num_spikes = 1, offset = 0, from_start = True, outside = True):
    offset = offset / arc_body.radius
    offset = (offset + 0.79 * size / arc_body.radius if from_start else -offset - 0.79 * size / arc_body.radius)
    for i in range(num_spikes):
        spike_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        spike_body.position = arc_body.position
        spike_body.angle = (arc_body.start_angle if from_start else arc_body.end_angle)
        spike_body.origin = [spike_body.position, spike_body.angle]
        spike_body.angular_velocity = arc_body.angular_velocity
        spike_body.geo_center = pymunk.Vec2d.from_polar(arc_body.radius + (arc_body.thickness/1.9 if outside else -arc_body.thickness/1.9), offset)
        base_coord_1 = pymunk.Vec2d.from_polar(size/2, offset + 1.5708) + spike_body.geo_center
        base_coord_2 = pymunk.Vec2d.from_polar(size/2, offset - 1.5708) + spike_body.geo_center
        peak_coord = pymunk.Vec2d.from_polar(size, offset + (0 if outside else 3.14159)) + spike_body.geo_center
        spike_shape = pymunk.Poly(spike_body, [base_coord_1, base_coord_2, peak_coord])
        spike_shape.collision_type = 2 # Spike collision
        space.add(spike_body, spike_shape)
        arc_body.spikes.append([spike_body, spike_shape])

        a = 0.8 if outside else 0.9
        offset = offset + (a * size / arc_body.radius if from_start else a * -size / arc_body.radius)
    return arc_body.spikes

def add_hinge(space: pymunk.Space, x: int, y: int, length: int, target: float, width = 5, angle = 0.0, speed = 1.5, left = True, centered = False):
    hinge_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    hinge_body.position = pymunk.Vec2d(x, y)
    hinge_body.angle = angle if left else 3.14159 - angle
    hinge_body.pivot_position = pymunk.Vec2d(x, y)
    hinge_body.geo_center = hinge_body.position if centered else (hinge_body.position + pymunk.Vec2d.from_polar(length, hinge_body.angle))
    hinge_body.width = width
    hinge_body.length = length
    hinge_body.target = target
    hinge_body.origin = hinge_body.angle
    hinge_body.speed = speed
    hinge_body.centered = centered
    hinge_body.left = left
    hinge_body.spikes = []
    hinge_body.direction = -1 if left else 1
    if centered:
        hinge_shape = pymunk.Poly.create_box(hinge_body, (length, width))
    else:
        hinge_shape = pymunk.Poly(hinge_body, [pymunk.Vec2d.from_polar(width/2, 1.5708), 
                                           pymunk.Vec2d.from_polar(width/2, -1.5708), 
                                           pymunk.Vec2d.from_polar(width/2, -1.5708) + pymunk.Vec2d(length, 0), 
                                           pymunk.Vec2d.from_polar(width/2, 1.5708) + pymunk.Vec2d(length, 0)])
    hinge_shape.collision_type = 6 # Hinge collision
    space.add(hinge_body, hinge_shape)
    return [[hinge_body, hinge_shape, False]]

def add_hinge_spike(space: pymunk.Space, hinge_shape: pymunk.Shape, size = 10, num_spikes = 1, offset = 0):
    base_coord_1 = pymunk.Vec2d.from_polar(size, -2.6180)
    base_coord_2 = pymunk.Vec2d.from_polar(size, -0.5236)
    peak_coord = pymunk.Vec2d.from_polar(size, 1.5708)

    for i in range(num_spikes):
        spike_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        spike_body.position = hinge_shape.body.position
        spike_body.geo_center = pymunk.Vec2d(hinge_shape.body.length / (2 if hinge_shape.body.centered else 1), hinge_shape.body.width/2) + pymunk.Vec2d.from_polar(size, (0.5236 if not hinge_shape.body.left else 2.6180)) + pymunk.Vec2d.from_polar(offset, 0) + (pymunk.Vec2d(-hinge_shape.body.length * (1 if hinge_shape.body.centered else 2), 0) if not hinge_shape.body.left else pymunk.Vec2d(0, 0))
        spike_body.angle = (hinge_shape.body.angle if hinge_shape.body.left else 3.14159 - hinge_shape.body.angle)
        spike_body.origin = [spike_body.position, spike_body.angle]
        spike_shape = pymunk.Poly(spike_body, [base_coord_1 + spike_body.geo_center, base_coord_2 + spike_body.geo_center, peak_coord + spike_body.geo_center])
        spike_shape.collision_type = 2 # Spike collision 
        space.add(spike_body, spike_shape)
        hinge_shape.body.spikes.append([spike_body, spike_shape])

        offset -= (1.5 * size if hinge_shape.body.left else -1.5 * size)
    return hinge_shape.body.spikes

def add_start(space: pymunk.Space, x: int, width = 28, thickness = 5):
    left_gate_body = pymunk.Body(mass = 10, moment = 1, body_type=pymunk.Body.DYNAMIC)
    right_gate_body = pymunk.Body(mass = 10, moment = 1, body_type=pymunk.Body.DYNAMIC)
    left_gate_body.position = (x - width/4, 575)
    right_gate_body.position = (x + width/4, 575)
    left_gate_body.pivot_position = pymunk.Vec2d(x - width/2, 575)
    right_gate_body.pivot_position = pymunk.Vec2d(x + width/2, 575)
    left_gate_body.data = [left_gate_body.position, left_gate_body.angle]
    right_gate_body.data = [right_gate_body.position, right_gate_body.angle]
    left_gate_body.width = 10
    right_gate_body.width = 10
    left_gate_shape = pymunk.Poly.create_box(left_gate_body, (width/2, thickness))
    right_gate_shape = pymunk.Poly.create_box(right_gate_body, (width/2, thickness))
    left_gate_joint = pymunk.PivotJoint(space.static_body, left_gate_body, left_gate_body.pivot_position)
    right_gate_joint = pymunk.PivotJoint(space.static_body, right_gate_body, right_gate_body.pivot_position)
    space.add(left_gate_body, right_gate_body, left_gate_shape, right_gate_shape, left_gate_joint, right_gate_joint)
    return [left_gate_body, right_gate_body, left_gate_shape, right_gate_shape, left_gate_joint, right_gate_joint]

def random_world(space: pymunk.Space, platforms, spikes, conveyors, goal, arcs, hinges, start):
    x = random.uniform(50, 250)
    y = 550
    start.extend(add_start(space, x))
    y -= random.randint(25, 50)
    next_obj = random.randint(0, 1)
    while y > 100:
        
        if next_obj == 0:
            left = True
            if x > 200:
                left = True
            elif x < 100:
                left = False
            else:
                left = random.uniform(-1, 1) > 0
            platforms.extend(add_platform(space, x, y, random.uniform(0.15, 0.25) * (1 if left else -1)))
            spikes.extend(add_platform_spike(space, platforms[-1][1], num_spikes=random.randint(0, 2)))
            x += random.uniform(70, 110) * (-1 if left else 1)
            y -= 75
        elif next_obj == 1:
            left = x > 150
            target = x + random.randint(70, 100) * (-1 if left else 1)
            conveyors.extend(add_conveyor(space, x, y, 100, target))
            spikes.extend(add_conveyor_spike(space, conveyors[-1][1], left=left, num_spikes=random.randint(0, 2)))
            x = target + random.randint(100, 125) * (-1 if left else 1)
            y -= 75
        elif next_obj == 2:
            left = True
            if x > 200:
                left = False
            elif x < 100:
                left = True
            else:
                left = random.uniform(-1, 1) > 0
            hinges.extend(add_hinge(space, x - (50 if left else -50), y, 100, -0.7854 if left else 3.92699, left=left))
            spikes.extend(add_hinge_spike(space, hinges[-1][1], num_spikes=random.randint(0, 2)))
            y -= 75
            x += random.randint(50, 75) * (1 if left else -1)
            
        elif next_obj == 3:
            y -= 50
            arcs.extend(add_arc(space, x, y, 50))
            spikes.extend(add_arc_spike(space, arcs[-1][0], num_spikes=random.randint(0, 2)))
            spikes.extend(add_arc_spike(space, arcs[-1][0], num_spikes=random.randint(0, 2), outside=False, offset=random.randint(0, 25)))
            y -= 100
        next_obj = random.randint(0, 3)
            
    goal.extend(add_goal(space, x, 50))
    
def map_4(space: pymunk.Space, platforms, spikes, conveyors, goal, arcs, hinges, start):
    start.extend(add_start(space, 150))
    platforms.extend(add_platform(space, 150, 500, 0.25))
    spikes.extend(add_platform_spike(space, platforms[0][1], num_spikes=2))
    conveyors.extend(add_conveyor(space, 200, 150, 100, 150))
    spikes.extend(add_conveyor_spike(space, conveyors[0][1], left=False, num_spikes=2))
    arcs.extend(add_arc(space, 175, 250, 50))
    spikes.extend(add_arc_spike(space, arcs[0][0], num_spikes=2))
    spikes.extend(add_arc_spike(space, arcs[0][0], num_spikes=2, outside=False, offset=25))
    goal.extend(add_goal(space, 50, 50))
    platforms.extend(add_platform(space, 83, 70, -0.5, width=90))
    hinges.extend(add_hinge(space, 0, 400, 100, -0.7854, left=True))
    spikes.extend(add_hinge_spike(space, hinges[0][1], num_spikes=2))

def map_2(space: pymunk.Space, platforms, spikes, conveyors, goal, arcs, hinges, start):
    start.extend(add_start(space, 50))
    conveyors.extend(add_conveyor(space, 50, 500, 100, 150))
    spikes.extend(add_conveyor_spike(space, conveyors[0][1], left=False, num_spikes=2))
    arcs.extend(add_arc(space, 250, 400, 50))
    spikes.extend(add_arc_spike(space, arcs[0][0], offset=25, num_spikes=2))
    spikes.extend(add_arc_spike(space, arcs[0][0], outside=False, offset=25))
    conveyors.extend(add_conveyor(space, 250, 300, 100, 150))
    spikes.extend(add_conveyor_spike(space, conveyors[1][1]))
    hinges.extend(add_hinge(space, 0, 150, 100, -0.7854))
    spikes.extend(add_hinge_spike(space, hinges[0][1]))
    goal.extend(add_goal(space, 200, 50))

def map_1(space: pymunk.Space, platforms, spikes, conveyors, goal, arcs, hinges, start):
    start.extend(add_start(space, 150))
    platforms.extend(add_platform(space, 150, 500, 0.25))
    platforms.extend(add_platform(space, 50, 400, -0.25))
    platforms.extend(add_platform(space, 250, 200, 0.5))
    platforms.extend(add_platform(space, 150, 100, -0.25))
    goal.extend(add_goal(space, 200, 50))

def map_3(space: pymunk.Space, platforms, spikes, conveyors, goal, arcs, hinges, start):
    start.extend(add_start(space, 250))
    arcs.extend(add_arc(space, 250, 425, 50, speed=-1.5))
    spikes.extend(add_arc_spike(space, arcs[0][0], num_spikes=8))
    spikes.extend(add_arc_spike(space, arcs[0][0], from_start=False, offset=100))
    spikes.extend(add_arc_spike(space, arcs[0][0], from_start=False, num_spikes=2, outside=False))
    platforms.extend(add_platform(space, 250, 325, 0))
    hinges.extend(add_hinge(space, 75, 200, 100, -0.5236))
    spikes.extend(add_hinge_spike(space, hinges[0][1]))
    conveyors.extend(add_conveyor(space, 250, 75, 100, 150))
    spikes.extend(add_conveyor_spike(space, conveyors[0][1]))
    goal.extend(add_goal(space, 50, 50))

def map_5(space: pymunk.Space, platforms, spikes, conveyors, goal, arcs, hinges, start):
    start.extend(add_start(space, 250))
    hinges.extend(add_hinge(space, 275, 550, 80, -0.7854, left=False))
    hinges.extend(add_hinge(space, 225, 500, 80, -0.7854, left=False))
    hinges.extend(add_hinge(space, 175, 450, 80, -0.7854, left=False))
    hinges.extend(add_hinge(space, 125, 400, 80, -0.7854, left=False))
    hinges.extend(add_hinge(space, 75, 350, 72, -0.7854, left=False))
    platforms.extend(add_platform(space, 0, 300, 1.5708))
    platforms.extend(add_platform(space, 0, 265, 0))
    platforms.extend(add_platform(space, 0, 280, -0.7854))
    platforms.extend(add_platform(space, 140, 385, 0.7854, width=10))
    platforms.extend(add_platform(space, 240, 485, 0.7854, width=10))
    platforms.extend(add_platform(space, 165, 360, 0.7854, 280))
    spikes.extend(add_platform_spike(space, platforms[-1][1], num_spikes=18))
    platforms.extend(add_platform(space, 300, 575, 1.5708))
    hinges.extend(add_hinge(space, 300, 300, 80, -0.25, left=False))
    spikes.extend(add_hinge_spike(space, hinges[-1][1]))
    arcs.extend(add_arc(space, 150, 150, 50, speed=-1.5))
    spikes.extend(add_arc_spike(space, arcs[0][0], from_start=False, num_spikes = 2))
    spikes.extend(add_arc_spike(space, arcs[0][0], from_start=False, outside=False, num_spikes=2))
    platforms.extend(add_platform(space, 150, 60, 0))
    goal.extend(add_goal(space, 50, 50))
