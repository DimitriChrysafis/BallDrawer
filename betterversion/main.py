import pygame
import numpy as np
import math
import random
import threading
import time
import json
import os
import hashlib
import cv2
import queue
import shutil
from physicsengine import update_positions, collision_detection
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
debugMode = False
haveBorders = False  # Set to True to draw thin black borders on the balls

screenWidth, screenHeight = (1920//1, 1080//1)
ballRadius = 4
cellSize = ballRadius * 2
gravity = 1000
subSteps = 8
baseDt = 1 / 60.0
dt = baseDt / subSteps
dt2 = dt * dt
numSpouts = 16
fixedAngle = math.radians(45)
fixedSpeed = 300.0
spoutStartX = (screenWidth - (numSpouts * 2 * ballRadius)) // 2 + ballRadius
spouts = [(spoutStartX + i * 2 * ballRadius, ballRadius) for i in range(numSpouts)]
spawnDelay = 0.001
ballArea = math.pi * (ballRadius ** 2)
screenArea = screenWidth * screenHeight
maxBalls = 3000000
simPositions = np.empty((maxBalls, 2), dtype=np.float32)
simPrev = np.empty((maxBalls, 2), dtype=np.float32)
radii = np.full(maxBalls, ballRadius, dtype=np.float32)
colors = []
renderPositionsOld = np.zeros((maxBalls, 2), dtype=np.float32)
renderPositionsCurrent = np.zeros((maxBalls, 2), dtype=np.float32)
cellsX = int(screenWidth // cellSize) + 1
cellsY = int(screenHeight // cellSize) + 1
lastSimTime = time.time()
simLock = threading.Lock()
nBalls = 0
originalBallCount = 0
mode = 0
coloringTriggered = False
mode1Colors = [None] * maxBalls
mode1SpawnIndex = 0
frameQueue = queue.Queue(maxsize=1000)
recordingActive = False
recordStop = False

def recordVideoThread():
    global recordStop
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 60, (screenWidth, screenHeight))
    while not recordStop or not frameQueue.empty():
        try:
            frame = frameQueue.get(timeout=0.1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        except queue.Empty:
            continue
    out.release()
    print("Recording thread finished and video file is finalized.")

def simulationLoop():
    global nBalls, mode, coloringTriggered, originalBallCount, recordStop, mode1SpawnIndex, recordingActive
    spawnTimer = 0.0
    fullnessThreshold = 0.99
    phase1_chill_start = None   # Settle timer for phase 1
    phase2_chill_start = None
    running = True
    pbar = None
    while running:
        try:
            simStart = time.time()
            spawnTimer += baseDt
            currentFullness = (nBalls * ballArea) / screenArea
            if mode == 0:
                if pbar is None:
                    pbar = tqdm(total=fullnessThreshold, desc="Phase 1: Filling", ncols=100, leave=True)
                pbar.n = currentFullness
                pbar.refresh()
                if currentFullness < fullnessThreshold and spawnTimer >= spawnDelay:
                    spawnTimer -= spawnDelay
                    for spout in spouts:
                        if nBalls < maxBalls and (nBalls * ballArea) / screenArea < fullnessThreshold:
                            x, y = spout
                            vx = fixedSpeed * math.cos(fixedAngle)
                            vy = fixedSpeed * math.sin(fixedAngle)
                            simPositions[nBalls] = (x, y)
                            simPrev[nBalls] = (x - vx * baseDt, y - vy * baseDt)
                            colors.append((255, 255, 255))
                            nBalls += 1
                # Wait 10 seconds for settling once fullness threshold is reached
                if currentFullness >= fullnessThreshold and not coloringTriggered:
                    if phase1_chill_start is None:
                        phase1_chill_start = time.time()
                    elif time.time() - phase1_chill_start >= 10:  # 10-second settle period
                        coloringTriggered = True
                        originalBallCount = nBalls
                        image = pygame.image.load("source_image.png").convert()
                        image = pygame.transform.scale(image, (screenWidth, screenHeight))
                        for i in range(nBalls):
                            x = int(np.clip(simPositions[i, 0], 0, screenWidth - 1))
                            y = int(np.clip(simPositions[i, 1], 0, screenHeight - 1))
                            c = image.get_at((x, y))
                            mode1Colors[i] = (c.r, c.g, c.b)
                            colors[i] = (c.r, c.g, c.b)
                        ballData = [{"color": c} for c in mode1Colors[:nBalls]]
                        with open("ball_data.json", "w") as f:
                            json.dump({"ball_count": nBalls, "balls": ballData}, f, indent=4)
                        nBalls = 0
                        colors.clear()
                        mode = 1
                        mode1SpawnIndex = 0
                        recordingActive = True
                        pbar.close()
                        pbar = None
                        print("Phase 1 reached 0.99 fullness and settled for 10 seconds. Color mapping complete. Starting phase 2 replay and video recording.")
            elif mode == 1:
                if pbar is None:
                    pbar = tqdm(total=originalBallCount * 0.99, desc="Phase 2: Replaying", ncols=100, leave=True)
                if spawnTimer >= spawnDelay:
                    spawnTimer -= spawnDelay
                    for i in range(numSpouts):
                        if mode1SpawnIndex < originalBallCount and nBalls < maxBalls:
                            x, y = spouts[i]
                            vx = fixedSpeed * math.cos(fixedAngle)
                            vy = fixedSpeed * math.sin(fixedAngle)
                            simPositions[nBalls] = (x, y)
                            simPrev[nBalls] = (x - vx * baseDt, y - vy * baseDt)
                            colors.append(mode1Colors[mode1SpawnIndex])
                            nBalls += 1
                            mode1SpawnIndex += 1
                pbar.n = nBalls
                pbar.refresh()
                if nBalls >= 0.99 * originalBallCount:
                    if phase2_chill_start is None:
                        phase2_chill_start = time.time()
                    elif time.time() - phase2_chill_start >= 10:  # 10-second settle period in phase 2
                        print("Phase 2 reached 0.99 of original ball count and settled for 10 seconds. Stopping simulation and recording.")
                        recordStop = True
                        running = False
                        pygame.event.post(pygame.event.Event(pygame.QUIT))
            for _ in range(subSteps):
                update_positions(simPositions, simPrev, radii, nBalls, dt, dt2, screenWidth, screenHeight, gravity, 0)
                collision_detection(simPositions, radii, nBalls, cellSize, cellsX, cellsY)
            with simLock:
                renderPositionsOld[:nBalls] = renderPositionsCurrent[:nBalls]
                renderPositionsCurrent[:nBalls] = simPositions[:nBalls]
                lastSimTime = time.time()
            simElapsed = time.time() - simStart
            sleepTime = baseDt - simElapsed
            if sleepTime > 0:
                time.sleep(sleepTime)
        except KeyboardInterrupt:
            running = False

pygame.init()
screen = pygame.display.set_mode((screenWidth, screenHeight))
pygame.display.set_caption("Adjacent Spouts Fluid Display")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 24)

def drawStatus():
    if mode == 0:
        fullness = (nBalls * ballArea) / screenArea * 100
    else:
        fullness = (nBalls / originalBallCount) * 100 if originalBallCount > 0 else 0
    status = [
        f"Mode: {'Filling' if mode == 0 else 'Replaying'}",
        f"Balls: {nBalls}/{maxBalls}",
        f"Fullness: {fullness:.2f}%",
        f"FPS: {clock.get_fps():.1f}"
    ]
    yOffset = 10
    for text in status:
        surface = font.render(text, True, (255, 255, 0), (0, 0, 0))
        screen.blit(surface, (10, yOffset))
        yOffset += 30

recordThread = threading.Thread(target=recordVideoThread)
recordThread.start()
simThread = threading.Thread(target=simulationLoop)
simThread.start()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((30, 30, 30))
    with simLock:
        nb = nBalls
        rOld = renderPositionsOld.copy()
        rCurrent = renderPositionsCurrent.copy()
        lastUpdate = lastSimTime
        colorsCopy = colors.copy()
    alpha = min((time.time() - lastUpdate) / baseDt, 1.0)
    interp = rOld[:nb] * (1 - alpha) + rCurrent[:nb] * alpha
    for i in range(nb):
        x, y = map(int, interp[i])
        # Skip drawing if the ball is still at a spout (using a 2-pixel tolerance)
        if abs(y - ballRadius) < 2 and any(abs(x - sp[0]) < 2 for sp in spouts):
            continue
        col = colorsCopy[i] if i < len(colorsCopy) else (255, 255, 255)
        if haveBorders:
            border_width = 1  # Thin border width
            pygame.draw.circle(screen, (0, 0, 0), (x, y), ballRadius)
            pygame.draw.circle(screen, col, (x, y), ballRadius - border_width)
        else:
            pygame.draw.circle(screen, col, (x, y), ballRadius)
    drawStatus()
    pygame.display.flip()
    clock.tick(60)
    if recordingActive:
        frame = pygame.surfarray.array3d(pygame.display.get_surface())
        frame = np.transpose(frame, (1, 0, 2))
        try:
            frameQueue.put_nowait(frame)
        except queue.Full:
            pass
simThread.join()
recordThread.join()
pygame.quit()
with open("output.mp4", "rb") as f:
    mp4Data = f.read()
mp4Hash = hashlib.md5(mp4Data).hexdigest()[:6]
with open("phase2_hash.txt", "w") as f:
    f.write(mp4Hash)
print("Final MP4 hash (first 6 hex digits):", mp4Hash)
folder_index = 1
while os.path.exists(str(folder_index)):
    folder_index += 1
folder_name = str(folder_index)
os.makedirs(folder_name)
new_video_name = os.path.join(folder_name, f"{mp4Hash}.mp4")
os.rename("output.mp4", new_video_name)
shutil.copy("source_image.png", folder_name)
print(f"Results saved in folder '{folder_name}' with video named '{mp4Hash}.mp4'.")
