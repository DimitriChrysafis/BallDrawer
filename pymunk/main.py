import os
import pygame
import pygame.gfxdraw
import pymunk
import math
import json
import random
import imageio
import numpy as np
import string
import shutil

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
BALL_SPEED = 1000
SETTLE_DELAY_MS = 3000
GRAVITY = 900
MASS = 0.1
DATA_FILENAME = "ball_data.json"
MIN_BALL_RADIUS = 3
MAX_BALL_RADIUS = 10
FULL_SCREEN_PERCENT = 1.3
FINAL_WAIT_MS = 5000
NUM_SPOUTS = 36
BORDER_THICKNESS = 1
OSCILLATION_AMPLITUDE_DEG = 40
OSCILLATION_PERIOD = 5.0

def compute_emission_angle(ballNumber, emitter_index):
    batch = ballNumber // NUM_SPOUTS
    emission_time = batch / 60.0
    baseAngle = 45 if emitter_index < NUM_SPOUTS//2 else 135
    variation = (ballNumber % NUM_SPOUTS - (NUM_SPOUTS // 2)) * 0.5
    osc_offset = OSCILLATION_AMPLITUDE_DEG * math.sin(2 * math.pi * emission_time / OSCILLATION_PERIOD)
    return baseAngle + variation + osc_offset

def runSimulationAndRecord(screen, clock, width, height, image_filename, video_writer=None):
    originalImage = pygame.image.load(image_filename).convert_alpha()
    imgWidth, imgHeight = originalImage.get_size()
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)
    floorBody = pymunk.Body(body_type=pymunk.Body.STATIC)
    floorShape = pymunk.Poly.create_box(floorBody, (width, 20))
    floorBody.position = (width/2, height-10)
    floorShape.friction = 1.0
    floorShape.elasticity = 0.4
    space.add(floorBody, floorShape)
    leftWall = pymunk.Segment(space.static_body, (5,0), (5,height), 5)
    rightWall = pymunk.Segment(space.static_body, (width-5,0), (width-5,height), 5)
    leftWall.friction = rightWall.friction = 1.0
    leftWall.elasticity = rightWall.elasticity = 0.4
    space.add(leftWall, rightWall)
    ceiling = pymunk.Segment(space.static_body, (5,5), (width-5,5), 5)
    ceiling.friction = 1.0
    ceiling.elasticity = 0.4
    space.add(ceiling)
    emitter_positions = [(int((i+0.5)*width/NUM_SPOUTS), 50) for i in range(NUM_SPOUTS)]
    ballShapes = []
    ballCount = 0
    settleTimer = None
    simulationDone = False
    finalWaitStart = None
    accumulated_area = 0.0
    window_area = width * height
    fill_threshold = FULL_SCREEN_PERCENT * window_area
    simulation_time = 0.0
    while True:
        dt = 1.0/60.0
        simulation_time += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        if not simulationDone:
            if accumulated_area < fill_threshold:
                for i, emitter in enumerate(emitter_positions):
                    angleDeg = compute_emission_angle(ballCount, i)
                    angleRad = math.radians(angleDeg)
                    vx = BALL_SPEED * math.cos(angleRad)
                    vy = BALL_SPEED * math.sin(angleRad)
                    radius = random.randint(MIN_BALL_RADIUS, MAX_BALL_RADIUS)
                    accumulated_area += math.pi * (radius**2)
                    inertia = pymunk.moment_for_circle(MASS, 0, radius)
                    body = pymunk.Body(MASS, inertia)
                    body.position = emitter
                    body.velocity = (vx, vy)
                    shape = pymunk.Circle(body, radius)
                    shape.friction = 0.5
                    shape.elasticity = 0.4
                    shape.color = (0,0,255,255)
                    shape.ballNumber = ballCount
                    shape.ballRadius = radius
                    space.add(body, shape)
                    ballShapes.append(shape)
                    ballCount += 1
            else:
                if settleTimer is None:
                    settleTimer = pygame.time.get_ticks()
                elif pygame.time.get_ticks() - settleTimer > SETTLE_DELAY_MS and not simulationDone:
                    simulationDone = True
                    finalWaitStart = pygame.time.get_ticks()
                    for shape in ballShapes:
                        pos = shape.body.position
                        relX = max(0, min(1, pos.x/width))
                        relY = max(0, min(1, pos.y/height))
                        imgX = int(relX*(imgWidth-1))
                        imgY = int(relY*(imgHeight-1))
                        totalR = totalG = totalB = count = 0
                        region = 1
                        for ix in range(imgX-region, imgX+region+1):
                            for iy in range(imgY-region, imgY+region+1):
                                if 0 <= ix < imgWidth and 0 <= iy < imgHeight:
                                    col = originalImage.get_at((ix,iy))
                                    totalR += col.r
                                    totalG += col.g
                                    totalB += col.b
                                    count += 1
                        avgColor = (totalR//count, totalG//count, totalB//count, 255) if count else (0,0,0,255)
                        shape.color = avgColor
                    ballData = []
                    for shape in ballShapes:
                        ballData.append({"ballNumber": shape.ballNumber, "color": list(shape.color), "radius": shape.ballRadius})
                    with open(DATA_FILENAME, "w") as f:
                        json.dump(ballData, f, indent=2)
        else:
            if finalWaitStart is not None and pygame.time.get_ticks() - finalWaitStart > FINAL_WAIT_MS:
                break
        space.step(dt)
        screen.fill((255,255,255))
        for shape in ballShapes:
            pos = shape.body.position
            x = int(round(pos.x))
            y = int(round(pos.y))
            r = int(round(shape.radius))
            if -32768 <= x <= 32767 and -32768 <= y <= 32767 and 0 <= r <= 32767:
                pygame.gfxdraw.filled_circle(screen, x, y, r, shape.color[:3])
                if BORDER_THICKNESS > 0:
                    pygame.draw.circle(screen, (0,0,0), (x,y), r, BORDER_THICKNESS)
        for i, emitter in enumerate(emitter_positions):
            ex, ey = emitter
            baseAngle = 45 if i < NUM_SPOUTS//2 else 135
            current_angle = baseAngle + OSCILLATION_AMPLITUDE_DEG * math.sin(2*math.pi*simulation_time/OSCILLATION_PERIOD)
            angleRad = math.radians(current_angle)
            spout_length = 30
            spout_width = 20
            tip = (ex + spout_length * math.cos(angleRad), ey + spout_length * math.sin(angleRad))
            half_width = spout_width / 2.0
            perp = (-math.sin(angleRad), math.cos(angleRad))
            base_left = (ex + half_width * perp[0], ey + half_width * perp[1])
            base_right = (ex - half_width * perp[0], ey - half_width * perp[1])
            pygame.draw.polygon(screen, (255,0,0), [base_left, base_right, tip])
            pygame.draw.circle(screen, (0,255,0), (int(round(tip[0])), int(round(tip[1]))), 5)
        pygame.display.flip()
        if video_writer is not None:
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1,0,2))
            video_writer.append_data(frame)
        clock.tick(60)
    pygame.quit()

def runRelaunchSimulation(screen, clock, width, height, video_writer):
    with open(DATA_FILENAME, "r") as f:
        ballData = json.load(f)
    totalBalls = len(ballData)
    space = pymunk.Space()
    space.gravity = (0, GRAVITY)
    floorBody = pymunk.Body(body_type=pymunk.Body.STATIC)
    floorShape = pymunk.Poly.create_box(floorBody, (width, 20))
    floorBody.position = (width/2, height-10)
    floorShape.friction = 1.0
    floorShape.elasticity = 0.4
    space.add(floorBody, floorShape)
    leftWall = pymunk.Segment(space.static_body, (5,0), (5,height), 5)
    rightWall = pymunk.Segment(space.static_body, (width-5,0), (width-5,height), 5)
    leftWall.friction = rightWall.friction = 1.0
    leftWall.elasticity = rightWall.elasticity = 0.4
    space.add(leftWall, rightWall)
    ceiling = pymunk.Segment(space.static_body, (5,5), (width-5,5), 5)
    ceiling.friction = 1.0
    ceiling.elasticity = 0.4
    space.add(ceiling)
    emitter_positions = [(int((i+0.5)*width/NUM_SPOUTS), 50) for i in range(NUM_SPOUTS)]
    ballShapes = []
    ballIndex = 0
    settleTimer = None
    simulationDone = False
    finalWaitStart = None
    simulation_time = 0.0
    while True:
        dt = 1.0/60.0
        simulation_time += dt
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        if ballIndex < totalBalls:
            for i in range(len(emitter_positions)):
                if ballIndex >= totalBalls:
                    break
                data = ballData[ballIndex]
                emitter_index = data["ballNumber"] % NUM_SPOUTS
                emitter = emitter_positions[emitter_index]
                angleDeg = compute_emission_angle(data["ballNumber"], emitter_index)
                angleRad = math.radians(angleDeg)
                vx = BALL_SPEED * math.cos(angleRad)
                vy = BALL_SPEED * math.sin(angleRad)
                radius = data["radius"]
                inertia = pymunk.moment_for_circle(MASS, 0, radius)
                body = pymunk.Body(MASS, inertia)
                body.position = emitter
                body.velocity = (vx, vy)
                shape = pymunk.Circle(body, radius)
                shape.friction = 0.5
                shape.elasticity = 0.4
                shape.color = tuple(data["color"])
                shape.ballNumber = data["ballNumber"]
                shape.ballRadius = radius
                space.add(body, shape)
                ballShapes.append(shape)
                ballIndex += 1
        else:
            if settleTimer is None:
                settleTimer = pygame.time.get_ticks()
            elif not simulationDone and pygame.time.get_ticks() - settleTimer > SETTLE_DELAY_MS:
                simulationDone = True
                finalWaitStart = pygame.time.get_ticks()
                for shape in ballShapes:
                    shape.body.velocity = (0,0)
                    shape.body.angular_velocity = 0
        if not simulationDone:
            space.step(dt)
        else:
            if finalWaitStart is not None and pygame.time.get_ticks() - finalWaitStart > FINAL_WAIT_MS:
                break
        screen.fill((255,255,255))
        for shape in ballShapes:
            pos = shape.body.position
            x = int(round(pos.x))
            y = int(round(pos.y))
            r = int(round(shape.radius))
            if -32768 <= x <= 32767 and -32768 <= y <= 32767 and 0 <= r <= 32767:
                pygame.gfxdraw.filled_circle(screen, x, y, r, shape.color[:3])
                if BORDER_THICKNESS > 0:
                    pygame.draw.circle(screen, (0,0,0), (x,y), r, BORDER_THICKNESS)
        for i, emitter in enumerate(emitter_positions):
            ex, ey = emitter
            baseAngle = 45 if i < NUM_SPOUTS//2 else 135
            current_angle = baseAngle + OSCILLATION_AMPLITUDE_DEG * math.sin(2*math.pi*simulation_time/OSCILLATION_PERIOD)
            angleRad = math.radians(current_angle)
            spout_length = 30
            spout_width = 20
            tip = (ex + spout_length * math.cos(angleRad), ey + spout_length * math.sin(angleRad))
            half_width = spout_width / 2.0
            perp = (-math.sin(angleRad), math.cos(angleRad))
            base_left = (ex + half_width * perp[0], ey + half_width * perp[1])
            base_right = (ex - half_width * perp[0], ey - half_width * perp[1])
            pygame.draw.polygon(screen, (255,0,0), [base_left, base_right, tip])
            pygame.draw.circle(screen, (0,255,0), (int(round(tip[0])), int(round(tip[1]))), 5)
        pygame.display.flip()
        frame = pygame.surfarray.array3d(screen)
        frame = np.transpose(frame, (1,0,2))
        video_writer.append_data(frame)
        clock.tick(60)
    pygame.quit()

def get_image_files():
    exts = ('.png','.jpg','.jpeg')
    return [f for f in os.listdir('.') if os.path.isfile(f) and f.lower().endswith(exts)]

def get_next_dir():
    i = 1
    while os.path.exists(str(i)):
        i += 1
    return str(i)

def process_image(image_file):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    runSimulationAndRecord(screen, clock, WINDOW_WIDTH, WINDOW_HEIGHT, image_file, video_writer=None)
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    video_hash = "".join(random.choices(string.ascii_lowercase+string.digits, k=6))
    video_filename = f"{video_hash}.mp4"
    video_writer = imageio.get_writer(video_filename, fps=60)
    runRelaunchSimulation(screen, clock, WINDOW_WIDTH, WINDOW_HEIGHT, video_writer)
    video_writer.close()
    if os.path.exists(DATA_FILENAME):
        os.remove(DATA_FILENAME)
    dest_dir = get_next_dir()
    os.makedirs(dest_dir)
    shutil.move(image_file, os.path.join(dest_dir, image_file))
    shutil.move(video_filename, os.path.join(dest_dir, video_filename))

def main():
    for image_file in get_image_files():
        process_image(image_file)

if __name__ == "__main__":
    main()
