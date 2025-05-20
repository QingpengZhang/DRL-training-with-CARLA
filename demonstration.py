## human demonstration with joystick for agent training
# store the human demonstration in the buffer
import sys

sys.path.append('D:/02_Envs/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.15-py3.7-win-amd64.egg')
sys.path.append('D:/02_Envs/CARLA_0.9.15/WindowsNoEditor/PythonAPI/carla/')

import carla
import pygame
import numpy as np
import os
import time

import carla
import pygame
import numpy as np
import os
import time

# --- CARLA Setup ---
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
original_settings = world.get_settings()
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05 # 20 FPS
world.apply_settings(settings)
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

def process_img(image):
    #image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    #array = np.array(image.raw_data)
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]  #去掉alpha通道
    array = array[:, :, ::-1]  #RGB to BGR
    global surface1
    surface1 = pygame.surfarray.make_surface(array.swapaxes(0, 1))

# Survelliance Camera for RGB
surve_camera_bp = blueprint_library.find('sensor.camera.rgb')
surve_camera_bp.set_attribute('image_size_x', '1000')
surve_camera_bp.set_attribute('image_size_y', '1100')
#surve_camera_bp.set_attribute('sensor_tick', '0.05')  
surve_camera_transform = carla.Transform(carla.Location(x=-5, y=0, z=5), carla.Rotation(pitch=12, yaw=0, roll=0))
surve_camera = world.spawn_actor(surve_camera_bp, surve_camera_transform, attach_to=vehicle, attachment_type=carla.AttachmentType.SpringArm)
surve_camera.listen(lambda image: process_img(image))

# Camera Sensor for 
IMG_WIDTH, IMG_HEIGHT = 640, 480
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1000')
camera_bp.set_attribute('image_size_y', '1100')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
image_queue = [] # Using a simple list for recent image
camera.listen(lambda image: image_queue.append(image) if len(image_queue) < 2 else image_queue.pop(0) or image_queue.append(image))


# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((2000, 1100))
pygame.display.set_caption("Human Driving Data Collector")
joystick = pygame.joystick.Joystick(0)
joystick.init()

pygame.font.init()
font = pygame.font.Font('freesansbold.ttf', 32)

clock = pygame.time.Clock()

# --- Data Storage ---
DATA_PATH = "human_driving_data"
IMAGES_PATH = os.path.join(DATA_PATH, "images")
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)
actions_log = open(os.path.join(DATA_PATH, "actions.csv"), "w")
actions_log.write("timestamp,image_file,throttle,steer,brake\n")
frame_id = 0
FULLBLACK = (255, 255, 255)
FULLGREEN = (0, 255, 0)
FULLRED = (255, 0, 0)

try:
    running = True
    while running:
        world.tick() # Advance CARLA simulation

        # Get latest image from CARLA
        if not image_queue:
            clock.tick(20) # Match CARLA FPS
            continue
        carla_image = image_queue[-1] # Get the most recent image

        # Process CARLA image for Pygame display and data storage
        img_bgra = np.frombuffer(carla_image.raw_data, dtype=np.uint8).reshape((1100, 1000, 4))
        img_bgr = img_bgra[:, :, :3] # Discard alpha
        img_rgb_display = img_bgr[:, :, ::-1].copy() # BGR to RGB for Pygame display
        img_to_save = img_bgr # Save as BGR (OpenCV standard) or convert as needed

        # Pygame display
        surface2 = pygame.image.frombuffer(img_rgb_display.tobytes(), (1000, 1100), "RGB")
        screen.blit(surface1, (0, 0))
        screen.blit(surface2, (1000, 0))

        
        text_surface1 = font.render('Surveillance Camera', True, FULLBLACK, FULLGREEN)
        text_surface1_rect = text_surface1.get_rect()
        text_surface1_rect.center = (720, 60)
        text_surface2 = font.render('Data collector Camera', True, FULLBLACK, FULLRED)
        text_surface2_rect = text_surface2.get_rect()
        text_surface2_rect.center = (1720, 60)
        screen.blit(text_surface1, text_surface1_rect)
        screen.blit(text_surface2, text_surface2_rect)

        # Pygame event handling for quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get human input from Pygame with joystick
        throttle = joystick.get_axis(1)
        brake = joystick.get_axis(2)
        throttle = max(0.0, (throttle + 1) / 2)  # 转换到0-1范围
        brake = max(0.0, (brake + 1) / 2)
    
        # 转向处理（带死区过滤）
        steer = joystick.get_axis(0)
        if abs(steer) < 0.1:
            steer = 0.0
        # Add more controls (reverse, handbrake) if needed

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        vehicle.apply_control(control)

        # Save data
        current_timestamp = time.time()  #生成的时间戳形式为浮点数，如1598767892.345678，代表秒数
        image_filename = f"{frame_id:06d}.png" # Or .jpg
        # For PNG, use RGB: cv2.imwrite(os.path.join(IMAGES_PATH, image_filename), img_to_save[:,:,::-1])
        # For simplicity, let's assume Pygame can save it (or use a library like OpenCV)
        pygame.image.save(pygame.surfarray.make_surface(img_to_save.swapaxes(0,1)), os.path.join(IMAGES_PATH, image_filename))


        actions_log.write(f"{current_timestamp},{image_filename},{throttle},{steer},{brake}\n")
        frame_id += 1

        pygame.display.flip()
        clock.tick(20) # Match CARLA FPS

finally:
    print("Cleaning up...")
    actions_log.close()
    if camera: camera.destroy()
    if vehicle: vehicle.destroy()
    world.apply_settings(original_settings) # Revert to original settings
    pygame.quit()
    print("Data collection finished.")